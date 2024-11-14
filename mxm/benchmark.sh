
platform=0
if [ -n "$1" ]; then
    platform=$1
fi

echo "Using Platform Index: $platform"

platform=$1
logFolder="log_$(date +"%F_%H_%M_%S")"

mkdir -p $logFolder

echo "Directory created: $logFolder"

## mxm
echo "./mxm -p $platform -s 1024 -k mxm > $logFolder/mxm"
./mxm -p $platform -s 1024 -k mxm > $logFolder/mxm 
sleep 10

## mxmLI
echo "./mxm -p $platform -s 1024 -k mxmLI > $logFolder/mxmLI"
./mxm -p $platform -s 1024 -k mxmLI > $logFolder/mxmLI
sleep 10

## mxmLIfma
echo "./mxm -p $platform -s 1024 -k mxmLIfma > $logFolder/mxmLIfma"
./mxm -p $platform -s 1024 -k mxmLIfma > $logFolder/mxmLIfma
sleep 10

## mxmLIfmaUnroll
echo "./mxm -p $platform -s 1024 -k mxmLIfmaUnroll > $logFolder/mxmLIfmaUnroll"
./mxm -p $platform -s 1024 -k mxmLIfmaUnroll > $logFolder/mxmLIfmaUnroll
sleep 10

## mxmLIfmaUnroll with Thread Block = 16
echo "./mxm -p $platform -s 1024 -k mxmLIfmaUnroll -w 16 > $logFolder/mxmLIfmaUnrollBlock16"
./mxm -p $platform -s 1024 -k mxmLIfmaUnroll -w 16 > $logFolder/mxmLIfmaUnrollBlock16
sleep 10

