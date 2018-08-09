MODEL_CKPT=NN
OUTPUT_NODE_NAME=Output
INPUT_NODE_NAME=Input

runNN:
	python3 Script.py

.PHONY: freeze
freeze:
	@echo "\nFreezing model..." 
	(cd model; python3 /home/tensorflow/tensorflow/python/tools/freeze_graph.py \
             --input_graph=$(MODEL_CKPT).pb \
             --input_binary=true \
             --input_checkpoint=$(MODEL_CKPT).ckpt \
             --output_graph=$(MODEL_CKPT)_frozen.pb \
             --output_node_name=$(OUTPUT_NODE_NAME);)

.PHONY: compile
compile: freeze
	@echo "\nCompiling model to Movidius graph..."
	(cd model; mvNCCompile -s 12 $(MODEL_CKPT)_frozen.pb -in=$(INPUT_NODE_NAME) -on=$(OUTPUT_NODE_NAME);)

.PHONY: profile
profile: freeze
	@echo "\nProfiling the model..."
	(cd model; mvNCProfile -s 12 $(MODEL_CKPT)_frozen.pb -in=$(INPUT_NODE_NAME) -on=$(OUTPUT_NODE_NAME);)

.PHONY: check
check: freeze
	@echo "\nComparing results with standard TensorFlow..."
	(cd model; mvNCCheck -s 12 $(MODEL_CKPT)_frozen.pb -in=$(INPUT_NODE_NAME) -on=$(OUTPUT_NODE_NAME);)

.PHONY: run
run: compile
	@echo "\nRunning project..."
	(python3 run.py)

.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo "  make help - Shows this message.";
	@echo "  make freeze - Export the trained model for inference.";
	@echo "  make compile - Convert the trained model into Movidius graph file.";
	@echo "  make check - Compare inference results with that of TensorFlow running on CPU/GPU.";
	@echo "  make profile - Run the model on NCS and extract complexity, bandwidth and execution time for each layer.";

.PHONY: clean
clean:
	@echo "\nMaking clean...";
	rm -rf model
