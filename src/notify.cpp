
#include <map>
#include <stdexcept>
#include <sstream>
#include <atomic>
#include <limits>
#include <typeinfo>

#include <pvxs/log.h>

#include "p4p.h"

DEFINE_LOGGER(logme, "p4p.notify");

// special key to interrupt handle()
static
constexpr uint32_t interruptor(std::numeric_limits<uint32_t>::max());

namespace p4p {

Notifier::~Notifier() {}

struct NotifierImpl : public Notifier, public std::enable_shared_from_this<NotifierImpl> {
    const std::weak_ptr<NotificationHub::Pvt> weak_hub;

    // guarded by Pvt::lock
    bool ready = false;

    explicit
    NotifierImpl(std::weak_ptr<NotificationHub::Pvt> weak_hub)
        :weak_hub(weak_hub)
    {}
    virtual ~NotifierImpl();
    virtual void notify();
};

struct NotificationHub::Pvt : public std::enable_shared_from_this<NotificationHub::Pvt> {
    // const after create()
    const SOCKET tx;

    epicsMutex lock;

    // guarded by lock
    SOCKET rx;
    std::list<std::weak_ptr<NotifierImpl>> pending;
    std::list<NotifierImpl*> trash;

    // keep with Hub to ensure destruction from python side.
    // function may capture python objects.
    std::map<NotifierImpl*, std::function<void()>> actions;

    bool interrupted = false;

    Pvt(SOCKET tx, SOCKET rx);
    ~Pvt();
    void poke() const noexcept;
};

NotifierImpl::~NotifierImpl()
{
    if(auto hub = weak_hub.lock()) {
        bool wake = false;
        {
            Guard G(hub->lock);
            wake = hub->pending.empty() && hub->trash.empty();
            hub->trash.push_back(this);
        }
        if(wake)
            hub->poke();
    }
}

void NotifierImpl::notify()
{
    if(auto hub = weak_hub.lock()) {
        bool wake = false;
        {
            Guard G(hub->lock);
            if(!ready) {
                wake = hub->pending.empty() && hub->trash.empty();
                hub->pending.push_back(shared_from_this());
                ready = true;
            }
        }
        if(wake)
            hub->poke();
    }
}

NotificationHub NotificationHub::create(bool blocking)
{
    SOCKET s[2];
    compat_socketpair(s);

    NotificationHub ret;
    ret.pvt = std::make_shared<Pvt>(s[0], s[1]);
    // tx side always blocking.  Only need to send() when pending list becomes not empty
    if(!blocking) {
        compat_make_socket_nonblocking(ret.pvt->rx);
    }
    return ret;
}

void NotificationHub::close()
{
    if(pvt){
        Guard G(pvt->lock);

        if(pvt->rx!=INVALID_SOCKET) {
            epicsSocketDestroy(pvt->rx);
            pvt->rx = INVALID_SOCKET;
        }
        pvt->pending.clear();
        pvt->actions.clear();
    }
    pvt.reset();
}

SOCKET NotificationHub::fileno() const
{
    if(!pvt)
        throw std::invalid_argument("NULL NotificationHub");
    Guard G(pvt->lock);
    return pvt->rx;
}

std::shared_ptr<Notifier>
NotificationHub::add(std::function<void()>&& fn)
{
    if(!pvt)
        throw std::invalid_argument("NULL NotificationHub");

    Guard G(pvt->lock);

    auto ret(std::make_shared<NotifierImpl>(pvt->shared_from_this()));

    pvt->actions.emplace(ret.get(), std::move(fn));

    return ret;
}

std::shared_ptr<Notifier>
NotificationHub::add(PyObject *raw)
{
    auto handler(PyRef::borrow(raw));
    auto fn = [handler]() {
        PyLock L;
        auto ret(PyRef::allownull(PyObject_CallFunction(handler.obj, "")));
        if(!ret.obj) {
            PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
            PyErr_Print();
            PyErr_Clear();
        }
    };

    return add(fn);
}

void NotificationHub::handle() const
{
    if(!pvt)
        throw std::invalid_argument("NULL NotificationHub");

    Guard G(pvt->lock);
    SOCKET rx = pvt->rx;

    while(!pvt->interrupted) {
        constexpr size_t max_batch_size = 16u;
        char buf[max_batch_size];

        int ret;
        {
            UnGuard U(G);
            ret = recv(rx, buf, sizeof(buf), 0);
        }
        if(ret < 0) {
            auto err = SOCKERRNO;
            if(err == SOCK_EWOULDBLOCK || err == EAGAIN || err == SOCK_EINTR) {
                return; // try again later

            } else {
                std::ostringstream msg;
                msg<<__func__<<" Socket error "<<err;
                throw std::runtime_error(msg.str());
            }
        } else if(ret == 0) {
            throw std::logic_error("NotificationHub tx closed?");
        }
        // have at least one message

        poll();
    }

    pvt->interrupted = false;
}

void NotificationHub::poll() const
{
    Guard G(pvt->lock);

    // take ownership of TODO list now so that any concurrent additions
    // while unlocked will provoke a poke()
    auto trash(std::move(pvt->trash));
    auto pending(std::move(pvt->pending));

    for(auto notifee : trash) {
        auto it(pvt->actions.find(notifee));
        if(it!=pvt->actions.end()) {
            auto act(std::move(it->second));
            pvt->actions.erase(it);

            UnGuard U(G);

            act = nullptr;
        }
    }

    for(auto W : pending) {
        if(auto notifee = W.lock()) {
            if(!notifee->ready)
                continue;

            auto it(pvt->actions.find(notifee.get()));
            if(it==pvt->actions.end())
                continue;

            notifee->ready = false;

            try {
                UnGuard U(G);
                (it->second)();
            }catch(std::exception& e){
                log_err_printf(logme, "Unhandled exception in callback %s: %s",
                               it->second.target_type().name(),
                               e.what());
            }
        }
    }
}

void
NotificationHub::interrupt() const noexcept
{
    if(pvt) {
        {
            Guard G(pvt->lock);
            if(pvt->interrupted)
                return;
            pvt->interrupted = true;
        }
        char b = '!';
        auto ret = send(pvt->tx, &b, sizeof(b), 0);
        if(ret!=sizeof(b))
            log_warn_printf(logme, "%s unable to wakeup: %d,%d",
                            __func__, (int)ret, SOCKERRNO);
    }

}

NotificationHub::Pvt::Pvt(SOCKET tx, SOCKET rx)
    :tx(tx)
    ,rx(rx)
{}

NotificationHub::Pvt::~Pvt() {
    (void)epicsSocketDestroy(tx);
    if(rx!=INVALID_SOCKET)
        (void)epicsSocketDestroy(rx);
}

void NotificationHub::Pvt::poke() const noexcept
{
    char b = '!';
    auto ret = send(tx, &b, sizeof(b), 0);
    if(ret!=sizeof(b))
        log_warn_printf(logme, "%s unable to wakeup: %d,%d",
                        __func__, (int)ret, SOCKERRNO);
}

} // namespace p4p
