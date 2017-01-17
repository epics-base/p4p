TOP = ..
include $(TOP)/configure/CONFIG
include $(TOP)/configure/CONFIG_PY

# place .so in subdirectory
INSTALL_SHRLIB = $(PY_INSTALL_DIR)/p4p

LOADABLE_LIBRARY_HOST += _p4p

_p4p_SRCS += p4p_top.cpp
_p4p_SRCS += p4p_type.cpp
_p4p_SRCS += p4p_value.cpp
_p4p_SRCS += p4p_server.cpp
_p4p_SRCS += p4p_server_provider.cpp


_p4p_LIBS += pvAccess pvData Com

PY += p4p/__init__.py
PY += p4p/wrapper.py
PY += p4p/rpc.py
PY += p4p/server.py
PY += p4p/nt.py

PY += p4p/test/__init__.py
PY += p4p/test/test_type.py
PY += p4p/test/test_value.py

include $(TOP)/configure/RULES
include $(TOP)/configure/RULES_PY
