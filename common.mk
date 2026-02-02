# Copied from https://github.com/adepretis/docker-make-stub/blob/master/common.mk
BLUE      := $(shell tput -Txterm setaf 4)
GREEN     := $(shell tput -Txterm setaf 2)
TURQUOISE := $(shell tput -Txterm setaf 6)
WHITE     := $(shell tput -Txterm setaf 7)
YELLOW    := $(shell tput -Txterm setaf 3)
GREY      := $(shell tput -Txterm setaf 1)
RESET     := $(shell tput -Txterm sgr0)
RED       := $(shell tput -Txterm setaf 1)

SMUL      := $(shell tput smul)
RMUL      := $(shell tput rmul)

# Add the following 'help' target to your Makefile
# And add help text after each target name starting with '\#\#'
# A category can be added with @category
HELP_FUN = \
	%help; \
	use Data::Dumper; \
	while(<>) { \
		if (/^([_a-zA-Z0-9\-%]+)\s*:.*\#\#(?:@([a-zA-Z0-9\-_\s]+))?\t(.*)$$/ \
			|| /^([_a-zA-Z0-9\-%]+)\s*:.*\#\#(?:@([a-zA-Z0-9\-]+))?\s(.*)$$/) { \
			$$c = $$2; $$t = $$1; $$d = $$3; \
			push @{$$help{$$c}}, [$$t, $$d, $$ARGV] unless grep { grep { grep /^$$t$$/, $$_->[0] } @{$$help{$$_}} } keys %help; \
		} \
	}; \
	for (sort keys %help) { \
		printf("${WHITE}%24s:${RESET}\n\n", $$_); \
		for (@{$$help{$$_}}) { \
			printf("%s%25s${RESET}%s  %s${RESET}\n", \
				( $$_->[2] eq "Makefile" || $$_->[0] eq "help" ? "${YELLOW}" : "${GREY}"), \
				$$_->[0], \
				( $$_->[2] eq "Makefile" || $$_->[0] eq "help" ? "${GREEN}" : "${GREY}"), \
				$$_->[1] \
			); \
		} \
		print "\n"; \
	}

# make
.DEFAULT_GOAL := help

# Variable wrapper
define defw
	custom_vars += $(1)
	$(1) ?= $(2)
	export $(1)
	shell_env += $(1)="$$($(1))"
endef

# Variable wrapper for hidden variables
define defw_h
	$(1) := $(2)
	shell_env += $(1)="$$($(1))"
endef

# Colorized output for control functions (info, warning, error)
define verbose_info
	@echo ${TURQUOISE}
	@echo ================================================================================
	@echo ${1}
	@echo ================================================================================
	@echo
	@tput -Txterm sgr0 # ${RESET} won't work here for some reason
endef

define verbose_warning
	@echo ${YELLOW}
	@echo ================================================================================
	@echo ${1}
	@echo ================================================================================
	@echo
	@tput -Txterm sgr0 # ${RESET} won't work here for some reason
endef

define verbose_error
	@echo ${RED}
	@echo ================================================================================
	@echo ${1}
	@echo ================================================================================
	@echo
	@tput -Txterm sgr0 # ${RESET} won't work here for some reason
endef

.PHONY: help
help:: ##@Usage show this help
#	@echo ""
#	@printf "%30s " "${BLUE}VARIABLES"
#	@echo "${RESET}"
#	@echo ""
	@printf "${BLUE}%25s${RESET}${TURQUOISE}  ${SMUL}%s${RESET}\n" $(foreach v, $(custom_vars), $v $(if $($(v)),$($(v)), ''))
	@echo ""
	@printf "%30s " "${TURQUOISE}TARGETS"
	@echo "${RESET}"
	@echo ""
	@perl -e '$(HELP_FUN)' $(MAKEFILE_LIST)
