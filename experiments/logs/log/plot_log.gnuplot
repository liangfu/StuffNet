# These snippets serve only as basic examples.
# Customization is a must.
# You can copy, paste, edit them in whatever way you want.
# Be warned that the fields in the training log may change in the future.
# You had better check the data files before designing your own plots.

# Please generate the neccessary data files with 
# /path/to/caffe/tools/extra/parse_log.sh before plotting.
# Example usage: 
#     ./parse_log.sh mnist.log
# Now you have mnist.log.train and mnist.log.test.
#     gnuplot mnist.gnuplot

# The fields present in the data files that are usually proper to plot along
# the y axis are test accuracy, test loss, training loss, and learning rate.
# Those should plot along the x axis are training iterations and seconds.
# Possible combinations:
# 1. Test accuracy (test score 0) vs. training iterations / time;
# 2. Test loss (test score 1) time;
# 3. Training loss vs. training iterations / time;
# 4. Learning rate vs. training iterations / time;
# A rarer one: Training time vs. iterations.

# What is the difference between plotting against iterations and time?
# If the overhead in one iteration is too high, one algorithm might appear
# to be faster in terms of progress per iteration and slower when measured
# against time. And the reverse case is not entirely impossible. Thus, some
# papers chose to only publish the more favorable type. It is your freedom
# to decide what to plot.

###### Fields in the data file your_log_name.log.train are
###### Iters Seconds TrainingLoss LearningRate

reset
set terminal png
set output "train_log.png"
set style data lines
set key right
set datafile separator ","

# Training loss vs. training iterations
set title "loss_cls vs. training iterations"
set xlabel "Training iterations"
set ylabel "loss_cls"
plot "faster_rcnn_end2end_VGG16_.txt.train" using 1:5 title "Training Characteristics"

# Training loss vs. training time
# plot "mnist.log.train" using 2:3 title "mnist"

# Learning rate vs. training iterations;
# plot "mnist.log.train" using 1:4 title "mnist"

# Learning rate vs. training time;
# plot "mnist.log.train" using 2:4 title "mnist"


###### Fields in the data file your_log_name.log.test are
###### Iters Seconds TestAccuracy TestLoss

# reset
# set terminal png
# set output "seg_finetune_val_log.png"
# set style data lines
# set key right
# 
# # Test loss vs. training iterations
# # plot "finetune_log.log.test" using 1:4 title "Val test accuracy characteristics"
# 
# # Test accuracy vs. training iterations
# set title "Val testing accuracy loss vs. training iterations"
# set xlabel "Training iterations"
# set ylabel "Val testing accuracy"
# plot "seg_finetune_log.log.test" using 1:3 title "Val test accuracy characteristics"

# Test loss vs. training time
# plot "mnist.log.test" using 2:4 title "mnist"

# Test accuracy vs. training time
# plot "mnist.log.test" using 2:3 title "mnist"
