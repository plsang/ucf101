function calker_build_train_test_splist()
	%% follow train/test splits from ICCV Workshop THUMOS
	split_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/iccv2013_splits.mat';
	if exist(split_file, 'file'),
		fprintf('File already exist! Skipped\n');
		return;
	end
	
	meta_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/metadata.mat';
	fprintf('load metadata...\n');
	metadata = load(meta_file, 'metadata');
	metadata = metadata.metadata;
	splits = {};
	
	% train/test 1
	test_idx = find(metadata.groups < 8);
	train_idx = find(metadata.groups >= 8);
	splits{1}.train_idx = train_idx;
	splits{1}.test_idx = test_idx;
	
	% train/test 2
	test_idx = find(metadata.groups > 7 & metadata.groups < 15);
	train_idx = find(metadata.groups <= 7 | metadata.groups >= 15);
	splits{2}.train_idx = train_idx;
	splits{2}.test_idx = test_idx;
	
	% train/test 3
	test_idx = find(metadata.groups > 14 & metadata.groups < 22);
	train_idx = find(metadata.groups <= 14 | metadata.groups >= 22);
	splits{3}.train_idx = train_idx;
	splits{3}.test_idx = test_idx;
	
	save(split_file, 'splits');
	
end
