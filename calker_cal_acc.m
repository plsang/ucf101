function calker_cal_acc(ker)
	
	%videolevel: 1 (default): video-based approach, 0: segment-based approach
	
	meta_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/metadata.mat';
	fprintf('--- Loading metadata...\n');
	metadata = load(meta_file, 'metadata');
	metadata = metadata.metadata;

	split_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/iccv2013_splits.mat';
	fprintf('--- Loading splits...\n');
	splits = load(split_file, 'splits');
	splits = splits.splits;
	

	% event names
	calker_exp_dir = sprintf('%s/%s/experiments/%s%s', ker.proj_dir, ker.proj_name, ker.feat, ker.suffix);
	
	fprintf('Scoring for feature %s...\n', ker.name);

	
	scorePath = sprintf('%s/scores/%s.scores.mat', calker_exp_dir, ker.name);
	
	accPath = sprintf('%s/scores/%s.accuracy.mat', calker_exp_dir, ker.name);
    
	if ~checkFile(scorePath), 
		error('File not found!! %s \n', scorePath);
	end
	
	load(scorePath, 'scores');

	n_class = length(metadata.all_classes);
	
	results = {};
	for ss = 1:length(splits),
	%for ss = 2:3,
		fprintf('Cal accuracy for split %d...\n', ss);
		
		split = splits{ss};
		
		split_scores = scores{ss};
		
		[~, predict_label] = max(split_scores);
		
		acc = zeros(n_class + 1, 1);
		for jj = 1:n_class,
			class_name = metadata.all_classes{jj};
			
			all_test_class_idx = metadata.classids(split.test_idx);
			
			test_class_idx = find(all_test_class_idx == jj);
			
			test_class_pre_label = predict_label(test_class_idx);
			
			acc(jj) = length(find(test_class_pre_label == jj))/length(test_class_idx);
			
		end
		
		acc(n_class + 1) = mean(acc(1:n_class));
		
		results{ss} = acc;
	end
	
	fprintf('Saving...\n');
	save(accPath, 'results');
	
end