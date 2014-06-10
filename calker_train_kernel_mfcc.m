function calker_train_kernel_mfcc(ker, start_split, end_split, start_class, end_class)

    test_on_train = 1;

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
 
	
	%load audio class
	audio_f = '/net/per610a/export/das11f/plsang/ucf101/metadata/audio_classes.txt';
	fh = fopen(audio_f, 'r');
	infos = textscan(fh, '%d %s');
	audio_classes_ids = infos{1};
	audio_classes_names = infos{2};
	fclose(fh);
	
	n_class = length(audio_classes_names);
	
	if ~exist('start_class', 'var'),
		start_class = 1;
	end
	
	if ~exist('end_class', 'var'),
		end_class = n_class;
	end
	 
	% indx of video that belong to audio classes
	audio_mask = ismember(metadata.classids, audio_classes_ids);
	
	kerPath = sprintf('%s/kernels/%s', calker_exp_dir, ker.devname);
	
	dist_Path = sprintf('%s.distmatrix.%s.mat', kerPath, ker.type);
	fprintf('--- Loading dist matrix...\n');
	kernels_ = load(dist_Path);
	
	selPath = sprintf('%s/kernels/%s.sel.mat', calker_exp_dir, ker.histName);
	fprintf('--- Loading sel matrix...\n');
	selected_idx = load(selPath, 'selected_idx');
	selected_idx = selected_idx.selected_idx;
	% mask audio idx using selected idx
	audio_mask = audio_mask' & selected_idx;
	
	for ss=start_split:end_split,
		
		split = splits{ss};
		
		model_dir = sprintf('%s/models/split-%d', calker_exp_dir, ss);		
		if ~exist(model_dir, 'file'),
			mkdir(model_dir);
		end
		
		% find index of class in audio classes only
		audio_idx = find(audio_mask > 0);
		audio_train_idx = intersect(split.train_idx, audio_idx);
		
		% base matrix 
		base = kernels_.matrix(audio_train_idx, audio_train_idx);
		
		all_labels = zeros(n_class, length(audio_train_idx));

		for ii = 1:length(audio_train_idx),
			idx = audio_train_idx(ii);
			for jj = 1:n_class,
				if metadata.classids(idx) == audio_classes_ids(jj),
					all_labels(jj, ii) = 1;
				else
					all_labels(jj, ii) = -1;
				end
			end
		end
	
		parfor kk = start_class:end_class,
			class_name = audio_classes_names{kk};
		
			modelPath = sprintf('%s/%s.%s.%s.model.mat', model_dir, class_name, ker.name, ker.type);
			
			if checkFile(modelPath),
				fprintf('Skipped training %s \n', modelPath);
				continue;
			end
			
			fprintf('Training class ''%s''...\n', class_name);	
			
			labels = double(all_labels(kk,:));
			posWeight = ceil(length(find(labels == -1))/length(find(labels == 1)));
			
			fprintf('SVM learning with predefined kernel matrix...\n');
			svm = calker_svmkernellearn(base, labels,   ...
							   'type', 'C',        ...
							   ...%'C', 10,            ...
							   'verbosity', 0,     ...
							   ...%'rbf', 1,           ...
							   'crossvalidation', 5, ...
							   'weights', [+1 posWeight ; -1 1]') ;

			svm = svmflip(svm, labels);
			
			% test it on train
			if test_on_train,		
				
				scores = svm.alphay' * base(svm.svind, :) + svm.b ;
				errs = scores .* labels < 0 ;
				err  = mean(errs) ;
				selPos = find(labels > 0) ;
				selNeg = find(labels < 0) ;
				werr = sum(errs(selPos)) * posWeight + sum(errs(selNeg)) ;
				werr = werr / (length(selPos) * posWeight + length(selNeg)) ;
				fprintf('\tSVM training error: %.2f%% (weighed: %.2f%%).\n', ...
				  err*100, werr*100) ;
				  
				% save model
				fprintf('\tNumber of support vectors: %d\n', length(svm.svind)) ;
				%clear kernels_;
			end
			
			fprintf('\tSaving model ''%s''.\n', modelPath) ;
			par_save_model( modelPath, svm );	

		end
	
	end
	
end

function par_save_model( modelPath, svm )
	ssave(modelPath, '-STRUCT', 'svm') ;
end
