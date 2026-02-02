.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

###########################
# VARIABLES
###########################
PROJECTNAME := 0cali-repr-simi
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD | tr / _)
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)

COMMA := ,
DASH := -
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)

# check if `netstat` is installed
ifeq (, $(shell which netstat))
$(error "Netstat executable not found, install it with `apt-get install net-tools`")
endif

# Check if Jupyter Port is already use and define an alternative
ifeq ($(origin PORT), undefined)
  PORT_USED = $(shell netstat -tl | grep -E '(tcp|tcp6)' | grep -Eo '8888' | tail -n 1)
  # Will fail if both ports 9999 and 10000 are used, I am sorry for that
  NEXT_TCP_PORT = $(shell netstat -tl | grep -E '(tcp|tcp6)' | grep -Eo '[0-9]{4}' | sort | tail -n 1 | xargs -I '{}' expr {} + 1)
  ifeq ($(PORT_USED), 8888)
    PORT = $(NEXT_TCP_PORT)
  else
    PORT = 8888
  endif
endif

# docker
ifeq ($(origin CONTAINER_NAME), undefined)
  CONTAINER_NAME := default
endif

ifeq ($(origin GPU_ID), undefined)
  GPU_ID := all
  GPU_NAME := $(GPU_ID)
else
  GPU_NAME = $(subst $(COMMA),$(DASH),$(GPU_ID))
endif

ifeq ("$(GPU)", "false")
  ifeq (, $(shell which nvidia-smi))
    GPU_ARGS := --shm-size 64G
  else
    GPU_ARGS := --gpus '"device="' --shm-size 64G
  endif
  DOCKER_CONTAINER_NAME := --name $(PROJECTNAME)_$(CONTAINER_NAME)
else
  GPU_ARGS := --gpus '"device=$(GPU_ID)"' --shm-size 64G --ipc=host
  DOCKER_CONTAINER_NAME := --name $(PROJECTNAME)_gpu_$(GPU_NAME)_$(CONTAINER_NAME)
endif

# count elements in comma-seperated GPU list
count = $(words $1)$(if $2,$(call count,$(wordlist 2,$(words $1),$1),$2))
GPU_LIST := $(subst $(COMMA),$(SPACE),$(GPU_ID))
NUM_GPUS := $(call count,$(GPU_LIST))

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    NUM_CORES := $(shell nproc)
else ifeq ($(UNAME_S),Darwin)
    NUM_CORES := $(shell sysctl -n hw.ncpu)
else
	NUM_CORES := 1
endif
# optimal number of threads is #cores/#gpus
NUM_THREADS := $(shell expr $(NUM_CORES) / $(NUM_GPUS))

DOCKER_ARGS := -v $$PWD:/workspace/ -p $(PORT):8888 --rm
DOCKER_CMD := docker run $(DOCKER_ARGS) $(GPU_ARGS) $(DOCKER_CONTAINER_NAME) -it $(PROJECTNAME):$(GIT_BRANCH)

###########################
# PROJECT UTILS
###########################
.PHONY: install
install:  ##@Utils install the dependencies for the project
	@python3 -m pip install -r requirements.txt
	@pre-commit install

.PHONY: clean
clean:  ##@Utils clean the project
	@black .
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store
	@rm -f .coverage coverage.xml report.xml
	@rm -f -R .pytest_cache
	@rm -f -R .idea
	@rm -f -R tmp/
	@rm -f -R cov_html/

###########################
# DOCKER
###########################
_build:
	@echo "Build image $(GIT_BRANCH)..."
	@docker build -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker run an interactive bash inside the docker image (default: GPU=true)
	@echo "Running bash with GPU being $(GPU) and GPU_ID $(GPU_ID)"
	$(DOCKER_CMD) /bin/bash; \

start_jupyter: _build  ##@Docker start a jupyter notebook inside the docker image
	@echo "Starting jupyter notebook"
	@-docker rm $(DOCKER_CONTAINER_NAME)
	$(DOCKER_CMD) /bin/bash -c "jupyter notebook --allow-root --ip 0.0.0.0 --port 8888"
