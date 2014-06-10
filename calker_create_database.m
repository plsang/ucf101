

function calker_create_database()
	video_dir = '/net/per610a/export/das11f/plsang/dataset/UCF101/video';
	list = dir(video_dir);
	
	% load class id from team
	id_f = '/net/per610a/export/das11f/plsang/dataset/UCF101/ucfTrainTestlist/classInd.txt';

	fh = fopen(id_f, 'r');
	infos = textscan(fh, '%d %s');
	fclose(fh);

	classIds = infos{1};
	classNames = infos{2};

	meta_file = '/net/per610a/export/das11f/plsang/ucf101/metadata/metadata.mat';
	if exist(meta_file, 'file'),
		fprintf('File already exist! Skipped\n');
		return;
	end
	
	metadata = struct;
	metadata.videos = {};
	metadata.classes = {};
	metadata.classids = [];
	metadata.groups = [];
	metadata.clips = [];
	
	all_classes = {};
	for ii = 1:length(list),
         
		if ~mod(ii, 1000),
			fprintf('%d ', ii);
		end
		
		file_name = list(ii).name;
		
		if strcmp(file_name, '.') || strcmp(file_name, '..'),
			continue;	
		end
		
		pattern = 'v_(?<class>\w+)_g(?<group>\d+)_c(?<clip>\d+).avi';
		
        info = regexp(file_name, pattern, 'names');
		if isempty(info), continue; end;
		
		video_id = file_name(1:end-4);
		
		metadata.videos = [metadata.videos; video_id];
		metadata.classes = [metadata.classes; info.class];
		
		class_id = find(ismember(classNames, info.class));
		if ~isempty(class_id),
			%% check
			metadata.classids = [metadata.classids; class_id];
		else
			error('Class not found [%s]', info.class);
		end
		
		metadata.groups = [metadata.groups; str2num(info.group)];
		metadata.clips = [metadata.clips; str2num(info.clip)];
	end
	
	metadata.classNames = classNames;
	save(meta_file, 'metadata');
end

