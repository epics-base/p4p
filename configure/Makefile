TOP=..

include $(TOP)/configure/CONFIG

ifeq ($(PY_VER),)
$(error Must set PY_VER to select a python version)
endif
PYTHON ?= python$(PY_VER)

TARGETS = $(CONFIG_TARGETS)

ifdef T_A
CONFIGS = CONFIG_PY RULES_PY os/CONFIG_PY$(PY_VER).Common.$(T_A)
endif

CONFIGS += $(subst ../,,$(wildcard $(CONFIG_INSTALLS)))

include $(TOP)/configure/RULES

os/CONFIG_PY$(PY_VER).Common.$(T_A): $(TOP)/makehelper.py
	[ -d $(dir $@) ] || $(MKDIR) $(dir $@)
	$(PYTHON) $< $@

ifdef BASE_3_15
realclean: py_clean
else
realclean:: py_clean
endif

py_clean:
	$(RMDIR) os
