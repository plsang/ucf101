function calker_test_kernel(ker, start_split, end_split) 


    % loading labels

	meta_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/metadata.mat';
	fprintf('--- Loading metadata...\n');
	metadata = load(meta_file, 'metadata');
	metadata = metadata.metadata;

	split_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/iccv2013_splits.mat';
	fprintf('--- Loading splits...\n');
	splits = load(split_file, 'splits');
	splits = splits.splits;

	if ~exist('start_split', 'var'),
		start_split = 1;
	end
	if ~exist('end_split', 'var'),
		end_split = length(splits);
	end

	calker_exp_dir = sprintf('%s/%s/experiments/%s%s', ker.proj_dir, ker.proj_name, ker.feat, ker.suffix);
 
	scorePath = sprintf('%s/scores/%s.scores.mat', calker_exp_dir, ker.name);
	
	if exist(scorePath, 'file'),
		fprintf('Score file exist!\n');
		return;
	end
 
    n_class = length(metadata.all_classes);

	kerPath = sprintf('%s/kernels/%s', calker_exp_dir, ker.devname);
	
	dist_Path = sprintf('%s.distmatrix.%s.mat', kerPath, ker.type);
	fprintf('--- Loading dist matrix...\n');
	kernels_ = load(dist_Path);
	
	scores = {};
	
	for ss=start_split:end_split,
		
		split = splits{ss};
		
		model_dir = sprintf('%s/models/split-%d', calker_exp_dir, ss);		
		if ~exist(model_dir, 'file'),
			mkdir(model_dir);
		end
		
		% base matrix 
		base = kernels_.matrix(split.train_idx, split.test_idx);
		
		all_labels = zeros(n_class, length(split.test_idx));

		for ii = 1:length(split.test_idx),
			idx = split.test_idx(ii);
			for jj = 1:n_class,
				if metadata.classids(idx) == jj,
					all_labels(jj, ii) = 1;
				else
					all_labels(jj, ii) = -1;
				end
			end
		end
		
		split_scores = zeros(n_class, length(split.test_idx));
		
		[N, Nt] = size(base) ;
		
		for jj = 1:n_class,
			class_name = metadata.all_classes{jj};
			
			labels = double(all_labels(jj,:));
			
			modelPath = sprintf('%s/%s.%s.%s.model.mat', model_dir, class_name, ker.name, ker.type);
			
			if ~checkFile(modelPath),
				error('Model not found %s \n', modelPath);			
			end
			
			fprintf('Loading model ''%s''...\n', class_name);
			class_model = load(modelPath);
			
			fprintf('-- [%d/%d] Testing class ''%s''...\n', jj, n_class, class_name);
			
			%[y, acc, dec] = svmpredict(zeros(Nt, 1), [(1:Nt)' base'], class_model.libsvm_cl, '-b 1') ;	
			[y, acc, dec] = svmpredict(labels', [(1:Nt)' base'], class_model.libsvm_cl, '-b 1') ;	
			fprintf('----- Accuracy: %f \n', acc);
			
			split_scores(jj, :) = dec(:, 1)';
		end
		
		scores{ss} = split_scores;
				
	end

		
	%saving scores
	fprintf('\tSaving scores ''%s''.\n', scorePath) ;
	save(scorePath, 'scores') ;
	
end