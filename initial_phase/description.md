# Premise

Initial epochs/steps of NN training often has much higher gradients comparing to later epochs. This can be a limiting factor for achieving higher accuracy at later epochs, especially with very large minibatches.

In this study I am recording gradients during first 10..100 steps and identifying a possible changes to initialization schema to allow reduced early stage gradients.

Related work: learning rate warm-up

Possible consequencies of changing initialization schema: necessity to adjust learning rate schedule.

Experiments: with updated init moch the gradient norms of regular training and compare final accuracies.

# Execution plan

Add recording callback to an existing NN implementation. Store gradients along with weights.

Visualize weight norms per layer, distribution of the gradients.