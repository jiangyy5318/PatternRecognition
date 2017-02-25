
#plot ".log.train" using 1:3 title "mnist"
#t version 5.0 demo of multiplot auto-layout capability
#
#
reset
set terminal png
set output "caffe.png"
set multiplot layout 3, 1 title "Multiplot layout 3, 1" font ",14"
set tmargin 2

set title "Training loss  vs. training iterations"
set xlabel "Training iterations"
set ylabel "Training loss "
plot "caffe.log.train" using 1:3 title "Trainloss"

set title "Test loss  vs. training iterations"
set xlabel "Training iterations"
set ylabel "Test Loss "
plot "caffe.log.test" using 1:4 title "Test loss"

set title "Test accuracy  vs. training iterations"
set xlabel "Training iterations"
set ylabel "Test Accuracy "
plot "caffe.log.test" using 1:3 title "Test Accuracy"



#
#set title "Plot 2"
#unset key
#plot 'caffe.log" using 1:2 ti 'silver.dat'
#
#set style histogram columns
#set style fill solid
#set key autotitle column
#set boxwidth 0.8
#set format y "    "
#set tics scale 0
#set title "Plot 3"
#plot 'immigration.dat' using 2 with histograms, \
#     '' using 7  with histograms , \
#     '' using 8  with histograms , \
#     '' using 11 with histograms 
#
#unset multiplot
#
#
#
