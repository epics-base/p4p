#cython: language_level=2

cdef extern from "<pvxs/log.h>" namespace "pvxs" nogil:
    enum Level:
        Debug "pvxs::Level::Debug"
        Info "pvxs::Level::Info"
        Warn "pvxs::Level::Warn"
        Err "pvxs::Level::Err"
        Crit "pvxs::Level::Crit"

    void logger_level_set(const char *name, int lvl) except+
    void logger_level_clear() except+
    void logger_config_env() except+
