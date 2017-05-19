TOP = ..
include $(TOP)/configure/CONFIG
include $(TOP)/configure/CONFIG_PY

# place .so in subdirectory
INSTALL_SHRLIB = $(PY_INSTALL_DIR)/p4p

LOADABLE_LIBRARY_HOST += _p4p

_p4p_SRCS += p4p_top.cpp
_p4p_SRCS += p4p_type.cpp
_p4p_SRCS += p4p_value.cpp
_p4p_SRCS += p4p_array.cpp
_p4p_SRCS += p4p_server.cpp
_p4p_SRCS += p4p_server_provider.cpp

_p4p_SRCS += p4p_client.cpp

_p4p_LIBS += pvAccess pvData Com

PY += p4p/__init__.py
PY += p4p/wrapper.py
PY += p4p/rpc.py
PY += p4p/server.py
PY += p4p/nt/__init__.py
PY += p4p/nt/common.py
PY += p4p/nt/scalar.py
PY += p4p/client/__init__.py
PY += p4p/client/cli.py
PY += p4p/client/raw.py
PY += p4p/client/thread.py

PY += p4p/test/__init__.py
PY += p4p/test/test_type.py
PY += p4p/test/test_value.py
PY += p4p/test/test_client.py
PY += p4p/test/test_rpc.py

include $(TOP)/configure/RULES
include $(TOP)/configure/RULES_PY

ifneq (,$(T_A))
nose:
	PYTHONPATH="$(abspath $(PWD))/python$(PY_LD_VER)/$(EPICS_HOST_ARCH)" $(PYTHON) -m nose.core -P p4p

# bounce back down to the sphinx generated Makefile
# aren't Makefiles fun...
sphinx:
	PYTHONPATH="$(abspath $(PWD))/python$(PY_LD_VER)/$(EPICS_HOST_ARCH)" $(MAKE) -C $(TOP)/documentation html

endif

.PHONY: nose sphinx
