TOP = ..
include $(TOP)/configure/CONFIG
include $(TOP)/configure/CONFIG_PY

# Our internal interfaces with generated cython
# are all c++, and MSVC doesn't allow extern "C" to
# return c++ types.
USR_CPPFLAGS += -D__PYX_EXTERN_C=extern

# place .so in subdirectory
INSTALL_SHRLIB = $(PY_INSTALL_DIR)/p4p

LOADABLE_LIBRARY_HOST += _p4p

_p4p_CPPFLAGS += -DPY_ARRAY_UNIQUE_SYMBOL=PVXS_PyArray_API
pvxs_value_CPPFLAGS += -DPY_ARRAY_UNIQUE_SYMBOL=PVXS_PyArray_API

_p4p_SRCS += _p4p.cpp
_p4p_SRCS += pvxs_type.cpp
_p4p_SRCS += pvxs_value.cpp
_p4p_SRCS += pvxs_sharedpv.cpp
_p4p_SRCS += pvxs_source.cpp
_p4p_SRCS += pvxs_client.cpp

_p4p_CPPFLAGS += -DPVXS_ENABLE_EXPERT_API

_p4p_LIBS += pvxs Com

LOADABLE_LIBRARY_HOST += _gw

_gw_CPPFLAGS += -DPVXS_ENABLE_EXPERT_API

_gw_SRCS += _gw.cpp
_gw_SRCS += pvxs_gw.cpp
_gw_SRCS += pvxs_odometer.cpp

_gw_LIBS += pvxs Com

PY += p4p/__init__.py
PY += p4p/disect.py
PY += p4p/wrapper.py
PY += p4p/rpc.py
PY += p4p/util.py
PY += p4p/version.py
PY += p4p/server/__init__.py
PY += p4p/server/raw.py
PY += p4p/server/thread.py
PY += p4p/server/cothread.py
PY += p4p/server/asyncio.py
PY += p4p/server/cli.py
PY += p4p/nt/__init__.py
PY += p4p/nt/common.py
PY += p4p/nt/scalar.py
PY += p4p/nt/enum.py
PY += p4p/nt/ndarray.py
PY += p4p/client/__init__.py
PY += p4p/client/cli.py
PY += p4p/client/raw.py
PY += p4p/client/thread.py
PY += p4p/client/asyncio.py
PY += p4p/client/cothread.py
PY += p4p/client/Qt.py

PY += p4p/gw.py
PY += p4p/asLib/__init__.py
PY += p4p/asLib/lex.py
PY += p4p/asLib/yacc.py
PY += p4p/asLib/pvlist.py

PY += p4p/test/__init__.py
PY += p4p/test/utils.py
PY += p4p/test/test_type.py
PY += p4p/test/test_value.py
PY += p4p/test/test_client_raw.py
PY += p4p/test/test_client_thread.py
PY += p4p/test/test_server.py
PY += p4p/test/test_rpc.py
PY += p4p/test/test_nt.py
PY += p4p/test/test_sharedpv.py
PY += p4p/test/test_gw.py
PY += p4p/test/test_asLib.py

PY += p4p/test/test_cothread.py
PY += p4p/test/cothreadtest.py

PY += p4p/test/test_qt.py
PY += p4p/test/qttest.py

PY += p4p/test/test_asyncio.py
PY += p4p/test/asynciotest.py

PY += p4p/example.conf
PY += p4p/pvagw@.service

PERL_SCRIPTS += pvagw

include $(TOP)/configure/RULES
include $(TOP)/configure/RULES_PY

ifdef T_A
pvagw: ../bootgw.py
	$(PYTHON) $< -P "../../python$(PY_LD_VER)/$(T_A)" $@
endif

_p4p.cpp: ../p4p/_p4p.pyx $(wildcard ../p4p/pvxs/*.pxd)
	$(PYTHON) -m cython -I ../p4p -o $@ --cplus $<

_p4p.h: _p4p.cpp ../p4p/_p4p.pyx

_gw.cpp: ../p4p/_gw.pyx $(wildcard ../p4p/pvxs/*.pxd)
	$(PYTHON) -m cython -I ../p4p -o $@ --cplus $<

_gw.h: _gw.cpp ../p4p/_gw.pyx

pvxs_gw$(OBJ): _gw.h
pvxs_gw$(DEP): _gw.h

pvxs_type$(DEP): _p4p.h
pvxs_type$(OBJ): _p4p.h

pvxs_value$(DEP): _p4p.h
pvxs_value$(OBJ): _p4p.h

pvxs_sharedpv$(DEP): _p4p.h
pvxs_sharedpv$(OBJ): _p4p.h

pvxs_source$(DEP): _p4p.h
pvxs_source$(OBJ): _p4p.h

pvxs_client$(DEP): _p4p.h
pvxs_client$(OBJ): _p4p.h

ifneq (,$(T_A))
nose.%:
	cd "$(abspath $(TOP))/python$(PY_LD_VER)/$(T_A)" && $(PYTHON) -m nose2 -v $* $(NOSEFLAGS)

# bounce back down to the sphinx generated Makefile
# aren't Makefiles fun...
sphinx:
	cd "$(abspath $(TOP))/python$(PY_LD_VER)/$(T_A)" && $(MAKE) -C $(TOP)/documentation html

sh:
	echo "export PYTHONPATH=\$${PYTHONPATH}:$(abspath $(TOP))/python$(PY_LD_VER)/$(T_A)" $(if $(OUTPUT),> $(OUTPUT))

ipython:
	PYTHONPATH="${PYTHONPATH}:$(abspath $(TOP))/python$(PY_LD_VER)/$(T_A)" $(PYTHON) -c "import sys; sys.argv[0] = '$(PYTHON)'; from IPython.terminal.ipapp import launch_new_instance; launch_new_instance()"

endif

.PHONY: nose sphinx sh
