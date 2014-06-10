meta_f = '/net/per610a/export/das11f/plsang/ucf101/metadata/metadata.mat';

load(meta_f, 'metadata');

% load class id from team
id_f = '/net/per610a/export/das11f/plsang/dataset/UCF101/ucfTrainTestlist/classInd.txt';

fh = fopen(id_f, 'r');
infos = textscan(fh, '%d %s');
fclose(fh);

classIds = infos{1};
classNames = infos{2};

for ii=1:length(classIds),
	meta_name = metadata.all_classes{ii};
	team_name = classNames{ii};
	
	if ~strcmp(meta_name, team_name),
		fprintf('[%d] not same: %s - %s \n', ii, meta_name, team_name);
	end
end
