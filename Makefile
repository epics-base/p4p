#Makefile at top of application tree
TOP = .
include $(TOP)/configure/CONFIG

DIRS += configure

DIRS += src
src_DEPEND_DIRS = configure

include $(TOP)/configure/RULES_TOP

UNINSTALL_DIRS += $(wildcard $(INSTALL_LOCATION)/python*)

# jump to a sub-directory where CONFIG_PY has been included
# can't include CONFIG_PY here as it may not exist yet
nose sphinx sh: all
	$(MAKE) -C src/O.$(EPICS_HOST_ARCH) $@

sphinx-clean:
	$(MAKE) -C documentation clean

sphinx-commit: sphinx
	touch documentation/_build/html/.nojekyll
	./commit-gh.sh documentation/_build/html

.PHONY: nose sphinx sphinx-commit sphinx-clean
