################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include objects.mk

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: CUDA

# Tool invocations
CUDA: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	/usr/local/cuda-10.1/bin/nvcc --cudart=static -L/usr/lib -ccbin g++ -gencode arch=compute_30,code=sm_30 -gencode arch=compute_30,code=compute_30 -o "CUDA" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS) CUDA
	-@echo ' '

.PHONY: all clean dependents

-include ../makefile.targets
