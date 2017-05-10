TOP=..

include $(TOP)/configure/CONFIG

PYTHON ?= python$(PY_VER)

TARGETS = $(CONFIG_TARGETS)

ifdef T_A
CONFIGS = CONFIG_PY RULES_PY os/CONFIG_PY.Common.$(T_A)
endif

CONFIGS += $(subst ../,,$(wildcard $(CONFIG_INSTALLS)))

include $(TOP)/configure/RULES

os/CONFIG_PY.Common.$(T_A): $(TOP)/makehelper.py
	[ -d $(dir $@) ] || $(MKDIR) $(dir $@)
	$(PYTHON) $< $@

ifdef BASE_3_15
realclean: py_clean
else
realclean:: py_clean
endif

py_clean:
	$(RMDIR) os
