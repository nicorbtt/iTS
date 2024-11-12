require("reticulate")
use_condaenv("its", required = TRUE)
py_config()

py_run_string("
import pickle as pkl

def read_pickle(dname):
  return pkl.load(open(f'toR/stacks_{dname}.pkl','rb'))
")

dname = "RAF"
x = py$read_pickle(dname)

titles = names(x)[1:5]
titles <- lapply(titles, function(x) { 
  n <- gsub("QL", "", x) 
  parse(text = paste0("bold(sQL[bold(", n, ")])"))  # TeX-style for subscript
})

par(mfrow = c(2,3), mar = c(4, 4, 2, 1), oma = c(0, 0, 0, 0))
for (i in 1:5) {
  tsutils::nemenyi(x[[i]], 
                   conf.level = 0.99, 
                   plottype = "vmcb",
                   labels = x$labels,
                   main=titles[[i]],
                   xlab = if (i >= 4) "Mean ranks" else "")
  
}
# mtext(dname, outer = TRUE, cex = 1, line = 1)

##############################################################
# a = rnorm(1000, mean=10, sd=1)
# b = rnorm(1000, mean=10.2, sd=1)
# x = matrix(data=c(a,b), ncol=2)
# tsutils::nemenyi(x, 
#                  conf.level = 0.95, 
#                  plottype = "vmcb",
#                  labels = c("a","b"),
#                  xlab = "Mean ranks")
