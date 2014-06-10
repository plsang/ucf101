function calker_cal_map_mfcc(ker)
	
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
	
	mapPath = sprintf('%s/scores/%s.map.mat', calker_exp_dir, ker.name);
    
	if ~checkFile(scorePath), 
		error('File not found!! %s \n', scorePath);
	end
	
	load(scorePath, 'scores');

        % load audio class
	audio_f = '/net/per610a/export/das11f/plsang/ucf101/metadata/audio_classes.txt';
	fh = fopen(audio_f, 'r');
	infos = textscan(fh, '%d %s');
	audio_classes_ids = infos{1};
	audio_classes_names = infos{2};
	fclose(fh);
	
	% indx of video that belong to audio classes
	audio_mask = ismember(metadata.classids, audio_classes_ids);
	
	n_class = length(audio_classes_names);

	results = {};
	for ss = 1:length(splits),
		fprintf('Cal accuracy for split %d...\n', ss);
		
		split = splits{ss};
		
		split_scores = scores{ss};
		
		audio_test_idx = find(audio_mask > 0);	% for all video, including not seletected
		audio_test_idx = intersect(split.test_idx, audio_test_idx);

		map = zeros(n_class + 1, 1);
		for jj = 1:n_class,
		
			this_scores = split_scores(jj, :);
			
			class_name = metadata.all_classes{jj};
			
			fprintf('Scoring for event [%s]...\n', class_name);
			
			[~, idx] = sort(this_scores, 'descend');
			
			all_test_class_idx = metadata.classids(audio_test_idx);
			
			audio_class_id = audio_classes_ids(jj);

			gt_idx = find(all_test_class_idx == audio_class_id);
			
			rank_idx = arrayfun(@(x)find(idx == x), gt_idx);
			
			sorted_idx = sort(rank_idx);	
			ap = 0;
			for kk = 1:length(sorted_idx), 
				ap = ap + kk/sorted_idx(kk);
			end
			ap = ap/length(sorted_idx);
			map(jj) = ap;
			
		end
		
		map(n_class + 1) = mean(map(1:n_class));
		
		results{ss} = map;
	end
	
	fprintf('Saving...\n');
	save(mapPath, 'results');
end
