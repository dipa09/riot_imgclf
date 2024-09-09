# This is the Makefile for the RIOT application. Copy this in the RIOT
# application directory e.g. RIOTBASE/examples/<app_name>/
#

# Application name, can be anything
APPLICATION = imgclf

# Select the board you are compiling for
#BOARD ?= native
#BOARD ?= arduino-mega2560
#BOARD ?= esp32-wroom-32

# This has to be the absolute path to the RIOT base directory:
RIOTBASE ?= $(CURDIR)/../..

# Comment this out to disable code in RIOT that does safety checking
# which is not needed in a production environment but helps in the
# development process:
DEVELHELP ?= 0

# Change this to 0 show compiler invocation lines by default:
QUIET ?= 1

# Modules used by 'main.c'
USEMODULE += ztimer_sec
USEMODULE += ztimer_usec
USEMODULE += benchmark

# Select which library to use. For emlearn you need to also copy it's source
# directory in the application root folder. So modify the path accordingly.
LIB=EMLEARN
ifeq ($(LIB),EMLEARN)
	CFLAGS += -I../emlearn
endif

# Tell 'main.c' that we are compiling on RIOT and which library have been used to
# compile the model.
CFLAGS += -DRIOT -D$(LIB)

# Disable all warnings
CFLAGS += -w

include $(RIOTBASE)/Makefile.include
