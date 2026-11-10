.PHONY: test pretrain grid bench ddp report all docker
test:      ; pytest -q
pretrain:  ; python run.py --stage pretrain
grid:      ; python run.py --stage grid
bench:     ; python run.py --stage bench
ddp:       ; python -m gfm.ddp --steps 100 && torchrun --nproc_per_node 2 -m gfm.ddp --steps 100
report:    ; python run.py --stage report
all:       ; python run.py
docker:    ; docker build -t genomic-fm-lab . && docker run --rm -v $(PWD)/results:/app/results genomic-fm-lab
