#Makefile at top of application tree
TOP = .
include $(TOP)/configure/CONFIG

DIRS += configure

DIRS += src
src_DEPEND_DIRS = configure

include $(TOP)/configure/RULES_TOP

UNINSTALL_DIRS += $(INSTALL_LOCATION)/python$(PY_VER)

nose:
	PYTHONPATH="$(PWD)/python$(PY_VER)/$(EPICS_HOST_ARCH)" python$(PY_VER) -m nose.core -P p4p

sphinx:
	PYTHONPATH="$(PWD)/python$(PY_VER)/$(EPICS_HOST_ARCH)" make -C documentation html

sphinx-commit: sphinx
	./commit-gh.sh documentation/_build/html

.PHONY: nose sphinx sphinx-commit
