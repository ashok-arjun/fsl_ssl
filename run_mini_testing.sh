for i in $(seq 1 1 1024)
do
    echo "Seed: $i"
    python save_features.py  --dataset miniImagenet --method protonet --amp --train_aug --seed $i 2>>err.log 1>>out.log
    python test.py  --dataset miniImagenet --method protonet --amp --train_aug --seed $i 2>>err.log 1>>out.log
done

