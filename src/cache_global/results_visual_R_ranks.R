require("reticulate")
use_condaenv("its", required = TRUE)
py_config()

py_run_string("
import pickle as pkl

def read_pickle(dname):
  return pkl.load(open(f'toR/stacks_{dname}.pkl','rb'))
")

dname = "Auto"
x = py$read_pickle(dname)

par(mfrow = c(1, 5), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0))
for (i in 1:5) {
  tsutils::nemenyi(x[[i]], 
                   conf.level = 0.99, 
                   plottype = "vmcb",
                   labels = x$labels,
                   main=names(x)[[i]])
}
mtext(dname, outer = TRUE, cex = 1, line = 1)