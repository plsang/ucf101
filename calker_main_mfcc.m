function calker_main_mfcc(feature_ext, feat_dim, ker_type, suffix, open_pool, start_split, end_split, start_class, end_class)

proj_name = 'ucf101';

addpath('/net/per900a/raid0/plsang/tools/kaori-secode-calker-v6/support');
addpath('/net/per900a/raid0/plsang/tools/libsvm-3.17/matlab');
addpath('/net/per900a/raid0/plsang/tools/vlfeat-0.9.16/toolbox');

% run vl_setup with no prefix
% vl_setup('noprefix');
vl_setup;


if ~exist('suffix', 'var'),
	suffix = '--calker-ucf';
end

if ~exist('feat_dim', 'var'),
	feat_dim = 4000;
end

if ~exist('ker_type', 'var'),
	ker_type = 'kl2';
end

if ~exist('open_pool', 'var'),
	open_pool = 0;
end

if ~exist('test_pat', 'var'),
	test_pat = 'kindredtest';
end

if isempty(strfind(suffix, '-ucf')),
	warning('**** Suffix does not contain ucf !!!!!\n');
end

ker = calker_build_kerdb(feature_ext, ker_type, feat_dim, suffix);
ker.proj_name = proj_name;
calker_exp_dir = sprintf('%s/%s/experiments/%s%s', ker.proj_dir, ker.proj_name, ker.feat, ker.suffix);
ker.log_dir = fullfile(calker_exp_dir, 'log');
 
%if ~exist(calker_exp_dir, 'file'),
mkdir(calker_exp_dir);
mkdir(fullfile(calker_exp_dir, 'metadata'));
mkdir(fullfile(calker_exp_dir, 'kernels'));
mkdir(fullfile(calker_exp_dir, 'scores'));
mkdir(fullfile(calker_exp_dir, 'models'));
mkdir(fullfile(calker_exp_dir, 'log'));
%end

calker_create_database();

%open pool
if matlabpool('size') == 0 && open_pool > 0, matlabpool(open_pool); end;
%calker_load_data(ker);
%calker_cal_kernel(ker);
%calker_train_kernel(ker, start_split, end_split, start_class, end_class);
%calker_test_kernel_mfcc(ker, start_split, end_split);
%calker_test_kernel(ker, start_split, end_split);
calker_cal_acc_mfcc(ker);
calker_cal_map_mfcc(ker);

%close pool
if matlabpool('size') > 0, matlabpool close; end;
