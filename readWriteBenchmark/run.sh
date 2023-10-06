

#        256MB     512MB    1024MB
sizes=( 67108864 134217728 268435456 )
offsets=( 0 16 20 24 128 )

for offset in "${offsets[@]}"
do
    for size in "${sizes[@]}"
    do
        echo "./read-write-benchmark -s $size -v $offset"
        ./read-write-benchmark -s $size -v $offset >> result_$offset.txt
    done
done