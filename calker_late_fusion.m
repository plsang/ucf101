function calker_late_fusion()

	proj_dir = '/net/per610a/export/das11f/plsang';
	proj_name = 'ucf101';
	
    suffix = '--calker-ucf';
	run_names = {'covdet.hessian.sift.cb256.devel.accumulate.pca.fc.l2',
				'densetrajectory.mbh.cb256.fc.pca.l2'};
				
	
	%fusion_name = 'fusion.video-based.global.local.motion.audio.trecvidmed10.devel.kcb.l2';
	fusion_name = 'fusion';
	for ii=1:length(run_names),
		fusion_name = sprintf('%s.%s', fusion_name, run_names{ii});
	end
	
	
	calker_exp_dir = sprintf('%s/%s/experiments/%s', proj_dir, proj_name, fusion_name);
	
	output_file = sprintf('%s/scores/%s.scores.mat', calker_exp_dir, fusion_name);
	output_dir = fileparts(output_file);
	
	if ~exist(output_dir, 'file'),
		mkdir(output_dir);
	end
	
	calker_exp_root = sprintf('%s/%s/experiments', proj_dir, proj_name);
	
	num_splits = 3;
	fused_scores = cell(1, num_splits);
	for ii=1:length(run_names),
		run_name = run_names{ii};
		score_file = sprintf('%s/%s%s/scores/%s.scores.mat', calker_exp_root, run_name, suffix, run_name);
		if ~exist(score_file, 'file'),
			error('File %s not found!\n', score_file);
		end
		load(score_file, 'scores');
		for jj = 1:num_splits,
			if isempty(fused_scores{jj}),
				fused_scores{jj} = scores{jj};
			else
				fused_scores{jj} = fused_scores{jj} + scores{jj};
				%fused_scores{jj} = max(fused_scores{jj}, scores{jj});
			end
		end
	end
	
	for jj = 1:num_splits,
		fused_scores{jj} = fused_scores{jj}./length(run_names);
	end
	
	scores = fused_scores;
	fprintf('Saving...\n');
	save(output_file, 'scores');
	
	acc_file = sprintf('%s/scores/%s.accuracy.mat', calker_exp_dir, fusion_name);
	
	meta_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/metadata.mat';
	fprintf('--- Loading metadata...\n');
	metadata = load(meta_file, 'metadata');
	metadata = metadata.metadata;

	split_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/iccv2013_splits.mat';
	fprintf('--- Loading splits...\n');
	splits = load(split_file, 'splits');
	splits = splits.splits;
	
	n_class = length(metadata.all_classes);
	
	results = {};
	for ss = 1:length(splits),
	%for ss = 1:2,
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
		
		acc(n_class + 1) = mean(acc);
		
		results{ss} = acc;
	end
	
	fprintf('Saving acc file...\n');
	acc_file
	save(acc_file, 'results');
	
	
	%%map
	
	mapPath = sprintf('%s/scores/%s.map.mat', calker_exp_dir, fusion_name);
	results = {};
	for ss = 1:length(splits),
		fprintf('Cal accuracy for split %d...\n', ss);
		
		split = splits{ss};
		
		split_scores = scores{ss};
		
		map = zeros(n_class + 1, 1);
		for jj = 1:n_class,
		
			this_scores = split_scores(jj, :);
			
			class_name = metadata.all_classes{jj};
			
			fprintf('Scoring for event [%s]...\n', class_name);
			
			[~, idx] = sort(this_scores, 'descend');
			
			all_test_class_idx = metadata.classids(split.test_idx);
			
			gt_idx = find(all_test_class_idx == jj);
			
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