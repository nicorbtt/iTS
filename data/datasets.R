library(expsmooth)
library(ZIM)
library(m5)

#before starting, set the working directory to the data folder
stopifnot(tail(strsplit(getwd(), c('/'))[[1]], 1) == 'data')

#carparts dataset
PATH_CARPARTS = file.path(getwd(), 'carparts')
if (!dir.exists(PATH_CARPARTS)) dir.create(PATH_CARPARTS)
carparts = carparts
path = file.path(PATH_CARPARTS, 'data_raw.csv')
write.csv2(carparts, path)

#syph dataset
PATH_SYPH = file.path(getwd(), 'syph')
if (!dir.exists(PATH_SYPH))dir.create(PATH_SYPH)
data("syph")
syph = syph[,3:69]
path = file.path(PATH_SYPH, 'data_raw.csv')
write.csv2(syph, path)

#M5 dataset
PATH_M5 = file.path(getwd(), 'M5')
if (!dir.exists(PATH_M5))dir.create(PATH_M5)
m5::m5_download(PATH_M5, unzip = TRUE)
to_delete = setdiff(list.files(PATH_M5, full.names = FALSE),
                    c('sales_train_evaluation.csv', 'sales_test_evaluation.csv',
                      'train.json', 'test.json', 'data.csv'))
file.remove(file.path(PATH_M5, to_delete))
