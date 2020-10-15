def stylize_axis(axis, grid=True):
    axis.set_facecolor('#eaeaf2')
    axis.set_axisbelow(True)
    if grid:
        axis.grid(color='w', linestyle='solid')
    
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)

def prepare_plt(plt, usetex=True):
	plt.rcParams.update({'font.size': 8})
	plt.rcParams.update({'text.usetex': usetex})
	# plt.rcParams.update({'text.latex.preamble': r'\renewcommand{\rmdefault}{ptm} \renewcommand{\sfdefault}{phv} \usepackage{amsfonts} '})
	plt.rcParams.update({'font.family': 'cm'})

	plt.rc('text', usetex=usetex)
	plt.rc('xtick', labelsize=7)
	plt.rc('ytick', labelsize=7)
	plt.rc('axes', labelsize=8)
	plt.rc('legend', fontsize=8)

PAD_INCHES = .1
# dimensions for AISTATS
WIDTH_2COLUMNS = 3.25
WIDTH_PAGE = 6.75

def fname_to_dict(fname):
 return dict(v.split('=') for v in fname.split(','))

filter_args = ['align_train', 'align_test',
			   'layer_align_train', 'layer_align_test',]
def dict_to_fname(args, filter_args=filter_args):
	name = ''
	for k, v in sorted(args.items(), key=lambda a: a[0]):
	    if k not in filter_args:
	        name += '%s=%s,' % (k, str(v))
	return name[:-1]