Run #   ---> Description

0       ---> Idle run
1-30    ---> Training with untouched data (200 epochs)
31-60   ---> Training with stragglers (identified at ep*=30) removed beforehand (200 epochs)
61-90   ---> Training with random data removed beforehand (200 epochs)
91-120  ---> Training with labels (training and test) reshuffled beforehand (200 epochs)

1001-1150 ---> Training with stragglers removed at consecutives ep*, starting from ep*=1 (200 epochs)
1151-1300 ---> Training with delta stragglers between ep* and ep*+1 removed at consecutives ep*, starting from ep*=1 (200 epochs)