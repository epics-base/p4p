TOP = ..
include $(TOP)/configure/CONFIG

PROD_HOST += odometer

odometer_SRCS += odometer.cpp

PROD_LIBS += pvxs
PROD_LIBS += $(EPICS_BASE_IOC_LIBS)

include $(TOP)/configure/RULES
