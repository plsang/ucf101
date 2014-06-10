function calker_test_kernel_mfcc(ker, start_split, end_split) 

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
 
    %load audio class
	audio_f = '/net/per610a/export/das11f/plsang/ucf101/metadata/audio_classes.txt';
	fh = fopen(audio_f, 'r');
	infos = textscan(fh, '%d %s');
	audio_classes_ids = infos{1};
	audio_classes_names = infos{2};
	fclose(fh);
	
		% indx of video that belong to audio classes
	audio_mask = ismember(metadata.classids, audio_classes_ids);
	
	n_class = length(audio_classes_names);

	kerPath = sprintf('%s/kernels/%s', calker_exp_dir, ker.devname);
	
	dist_Path = sprintf('%s.distmatrix.%s.mat', kerPath, ker.type);
	fprintf('--- Loading dist matrix...\n');
	kernels_ = load(dist_Path);
	
	selPath = sprintf('%s/kernels/%s.sel.mat', calker_exp_dir, ker.histName);
	fprintf('--- Loading sel matrix...\n');
	selected_idx = load(selPath, 'selected_idx');
	selected_idx = selected_idx.selected_idx;
	% mask audio idx using selected idx
	train_audio_mask = audio_mask' & selected_idx;
	
	scores = {};
	
	for ss=start_split:end_split,
		
		split = splits{ss};
		
		model_dir = sprintf('%s/models/split-%d', calker_exp_dir, ss);		
		if ~exist(model_dir, 'file'),
			mkdir(model_dir);
		end
		
		% find index of class in audio classes only
		audio_train_idx = find(train_audio_mask > 0);
		audio_train_idx = intersect(split.train_idx, audio_train_idx);
		
		audio_test_idx = find(audio_mask > 0);	% for all video, including not seletected
		audio_test_idx = intersect(split.test_idx, audio_test_idx);
		
		% base matrix 
		base = kernels_.matrix(audio_train_idx, audio_test_idx);
		
		all_labels = zeros(n_class, length(audio_test_idx));

		for ii = 1:length(audio_test_idx),
			idx = audio_test_idx(ii);
			for jj = 1:n_class,
				if metadata.classids(idx) == audio_classes_ids(jj),
					all_labels(jj, ii) = 1;
				else
					all_labels(jj, ii) = -1;
				end
			end
		end
		
		split_scores = zeros(n_class, length(audio_test_idx));
		
		[N, Nt] = size(base) ;
		
		for jj = 1:n_class,
			class_name = audio_classes_names{jj};
			
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