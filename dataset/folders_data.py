train400_dir    = "./sets/Train400/"
valid68_dir     = "./sets/Val68/"
test12_dir      = "./sets/Set12/"

list_test_10synt = [('%02d_%02d'%(item,index), './sets/Noisy100/synt%02d_%02d.mat'%(item,index)) for item in range(1,11) for index in range(0,10)]
