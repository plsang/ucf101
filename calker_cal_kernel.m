
function calker_cal_kernel(ker)

	feature_ext = ker.feat;

	calker_exp_dir = sprintf('%s/%s/experiments/%s%s', ker.proj_dir, ker.proj_name, ker.feat, ker.suffix);

	kerPath = sprintf('%s/kernels/%s', calker_exp_dir, ker.devname);
	
	histPath = sprintf('%s/kernels/%s.mat', calker_exp_dir, ker.histName);
	
	dist_Path = sprintf('%s.distmatrix.%s.mat', kerPath, ker.type);
	
	if ~exist(dist_Path),
		fprintf('Loading hist data...\n');
		data = load(histPath, 'data');
		data = data.data;
		hists = data.hists;
		
		fprintf('\tCalculating dist matrix for %s ... \n', feature_ext) ;	
		%ker = calcKernel(ker, dev_hists);
		
		if strcmp(ker.type, 'kl2'),
			fprintf('Calculating kl2 kernel...\n');
			ker.matrix = hists' * hists ;
		elseif  strcmp(ker.type, 'echi2'),
			fprintf('Calculating echi2 kernel...\n');
			ker.matrix = vl_alldist2(hists, 'chi2') ;
		end	
		
		fprintf('\tSaving kernel ''%s''.\n', dist_Path) ;
		par_save( dist_Path, ker );
	else
		fprintf('Skipped calculating kernel [%s]...\n', dist_Path);
	end

end

function par_save( output_file, ker )
	ssave(output_file, '-STRUCT', 'ker', '-v7.3');
end
