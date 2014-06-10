function calker_late_fusion_mfcc()

	proj_dir = '/net/per610a/export/das11f/plsang';
	proj_name = 'ucf101';
	
    suffix = '--calker-ucf';
	run_names = {'covdet.hessian.sift.cb256.devel.accumulate.pca.fc.l2',
				'densetrajectory.mbh.cb256.fc.pca.l2',
				'mfcc.rastamat.cb256.fc.l2'};
				
    meta_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/metadata.mat';
	fprintf('--- Loading metadata...\n');
	metadata = load(meta_file, 'metadata');
	metadata = metadata.metadata;
    
	%load audio class
	audio_f = '/net/per610a/export/das11f/plsang/ucf101/metadata/audio_classes.txt';
	fh = fopen(audio_f, 'r');
	infos = textscan(fh, '%d %s');
	audio_classes_ids = infos{1};
	audio_classes_names = infos{2};
	fclose(fh);
	
    audio_mask = ismember(metadata.classids, audio_classes_ids);
    
    split_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/iccv2013_splits.mat';
	fprintf('--- Loading splits...\n');
	splits = load(split_file, 'splits');
	splits = splits.splits;
    
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
            split = splits{jj};
            audio_test_idx = find(audio_mask > 0);	% for all video, including not seletected
            audio_test_idx = intersect(split.test_idx, audio_test_idx);
            
            audio_test_mask = ismember(split.test_idx, audio_test_idx);
            
			if isempty(fused_scores{jj}),
				fused_scores{jj} = scores{jj};
            elseif ii==2,
                fused_scores{jj} = fused_scores{jj} + scores{jj};
            else
                
				%fused_scores{jj} = fused_scores{jj} + scores{jj};
				%fused_scores{jj}(audio_classes_ids, audio_test_mask > 0 ) = max(fused_scores{jj}(audio_classes_ids, audio_test_mask > 0), scores{jj});
                fused_scores{jj}(audio_classes_ids, audio_test_mask > 0 ) = (fused_scores{jj}(audio_classes_ids, audio_test_mask > 0) + scores{jj});
			end
		end
	end
	
	%for jj = 1:num_splits,
	%	fused_scores{jj} = fused_scores{jj}./length(run_names);
	%end
	
	fprintf('Saving...\n');
	save(output_file, 'fused_scores');
	
	acc_file = sprintf('%s/scores/%s.accuracy.mat', calker_exp_dir, fusion_name);

	
	n_class = length(metadata.all_classes);
	
	results = {};
	for ss = 1:length(splits),
	%for ss = 1:2,
		fprintf('Cal accuracy for split %d...\n', ss);
		
		split = splits{ss};
		
		split_scores = fused_scores{ss};
		
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
	
	fprintf('Saving...\n');
ree	save(acc_file, 'results');
	
end