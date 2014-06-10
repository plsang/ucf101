
function calker_load_data_seb(ker)

%%Update change parameter to ker
% load database


%fea_dir = '/net/per610a/export/das11f/plsang/ucf101/feature/video';
fea_dir = '/net/per610a/export/das11f/ledduy/plsang/haipn/UCF101/descriptors';
meta_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/metadata.mat';

fprintf('load metadata...\n');
metadata = load(meta_file, 'metadata');
metadata = metadata.metadata;

videos = metadata.videos;



calker_exp_dir = sprintf('%s/%s/experiments/%s%s', ker.proj_dir, ker.proj_name, ker.feat, ker.suffix);

histPath = sprintf('%s/kernels/%s.mat', calker_exp_dir, ker.histName);
selPath = sprintf('%s/kernels/%s.sel.mat', calker_exp_dir, ker.histName);
if exist(histPath, 'file'),
	fprintf('Exist [%s]!\n', histPath);
	return;
end

data = struct;
hists = zeros(ker.num_dim, length(videos));
selected_idx = ones(1, length(videos));

parfor ii = 1:length(videos), %
	if mod(ii, 100) == 0,
		fprintf('%d/%d ... ', ii, length(videos));
	end
	
	video_name = videos{ii};
	%segment_path = sprintf('%s/%s/%s/%s.mat', fea_dir, ker.feat_raw, video_name, video_name);
	segment_path = sprintf('%s/%s/%s.mp4.%s', fea_dir, video_name, video_name, ker.feat_raw);
	
	if ~exist(segment_path),
		msg = sprintf('File [%s] does not exist!', segment_path);
		log(msg);
		code = zeros(ker.num_dim, 1);
	else
		%code = load(segment_path, 'code');
		%code = code.code;
		fh = fopen(segment_path);
		code = textscan(fh, '%f');
		code = code{:};
		fclose(fh);
	end
	
	if size(code, 1) ~= ker.num_dim,
		msg = sprintf('Dimension mismatch [%d-%d-%s]. Skipped !!', size(code, 1), ker.num_dim, segment_path);
		log(msg);
		code = zeros(ker.num_dim, 1);
		%continue;
	end
	
	if any(isnan(code)),
		msg = sprintf('Feature contains NaN [%s]. Skipped !!', segment_path);
		log(msg);
		%msg = sprintf('Feature contains NaN [%s]', segment_path);
		code = zeros(ker.num_dim, 1);
		%continue;
	end
	
	% event video contains all zeros --> skip, keep backgroud video
	if all(code == 0),
		%msg = sprintf('Feature contains all zeros [%s]. Skipped !!\n', segment_path);
		%log(msg);
		%continue;
		
		selected_idx(ii) = 0;
	end
	
	if ~all(code == 0),
		
		%code = sign(code) .* sqrt(abs(code));
		
		if strcmp(ker.feat_norm, 'l1'),
			code = code / norm(code, 1);
		elseif strcmp(ker.feat_norm, 'l2'),
			code = code / norm(code, 2);
		else
			error('unknown norm!\n');
		end
    end
	
	hists(:, ii) =  code;
	
end

data.hists = hists;
data.selected_idx = selected_idx;

fprintf('Saving data...\n');
save(histPath, 'data', '-v7.3');
save(selPath, 'selected_idx');

end


function log (msg)
    logfile = [mfilename('fullpath'), '.log'];
    fh = fopen(logfile, 'a+');
    fprintf(fh, [msg, '\n']);
	fclose(fh);
end
