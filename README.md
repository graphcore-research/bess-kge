# BESS for KGE in PopTorch

SRAM-only.

## Usage
1\. Install Poplar SDK 3.1.0 following the instructions in the Getting Started guide for your IPU system.

2\. Create a virtualenv and activate/install the required packages:
```
source $POPLAR_SDK_DIR/enable
python3.8 -m venv .venv
source .venv/bin/activate
pip install $POPLAR_SDK_DIR/poptorch_x.x.x.whl
source <path to poplar installation>/enable.sh
source <path to popart installation>/enable.sh
pip install -r requirements.txt 
```

3\. Build custom_ops.so with provided makefile:
```
```

4\. Run
```
```

## References

## License notice

