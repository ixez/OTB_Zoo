
function results=run_MEEM(seq, res_path, bSaveImage, args)
display = false;
init_rect = seq.init_rect;

clear global
addpath(genpath('.'));
% declare global variables
global sampler
global svm_tracker
global experts
global config
global finish % flag for determination by keystroke

sampler = createSampler();
svm_tracker = createSvmTracker();
experts = {};
finish = 0;

tic
for frame_id=1:length(seq.s_frames)    
    im=imread(seq.s_frames{frame_id});
    
    %% intialization
    if frame_id == 1
        config = makeConfig(im,init_rect,true,true,true,display);
        svm_tracker.output = init_rect*config.image_scale;
        svm_tracker.output(1:2) = svm_tracker.output(1:2) + config.padding;
        svm_tracker.output_exp = svm_tracker.output;
        output = svm_tracker.output;
    end
    
    %% compute ROI and scale image
    [I_scale]= getFrame2Compute(im);
    
    %% crop frame
    if frame_id == 1
        sampler.roi = rsz_rt(svm_tracker.output,size(I_scale),5*config.search_roi,false);
    else%if svm_tracker.confidence > config.svm_thresh
        sampler.roi = rsz_rt(output,size(I_scale),config.search_roi,true);
    end
    I_crop = I_scale(round(sampler.roi(2):sampler.roi(4)),round(sampler.roi(1):sampler.roi(3)),:);
    
    %% compute feature images
    [BC F] = getFeatureRep(I_crop,config.hist_nbin);
    
    %% tracking part
    if frame_id==1
        initSampler(svm_tracker.output,BC,F,config.use_color);
        train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
        label = sampler.costs(train_mask,1)<config.thresh_p;
        fuzzy_weight = ones(size(label));
        initSvmTracker(sampler.patterns_dt(train_mask,:), label, fuzzy_weight);
        
        if display
            figure(1);
            imshow(im);
            res = svm_tracker.output;
            res(1:2) = res(1:2) - config.padding;
            res = res/config.image_scale;
            rectangle('position',res,'LineWidth',2,'EdgeColor','b')
        end
    else
        % testing
        if display
            figure(1)
            imshow(im);       
            roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2)+1;
            roi_reg(1:2) = roi_reg(1:2) - config.padding;
            rectangle('position',roi_reg/config.image_scale,'LineWidth',1,'EdgeColor','r');
        end
        if mod(frame_id,config.expert_update_interval) == 0% svm_tracker.update_count >= config.update_count_thresh
            updateTrackerExperts;
        end

        expertsDo(BC,config.expert_lambda,config.label_prior_sigma);
        
        if svm_tracker.confidence > config.svm_thresh
            output = svm_tracker.output;
        end
        
        
        if config.display
            figure(1) 
            res = output;
            res(1:2) = res(1:2) - config.padding;
            res = res/config.image_scale;
            if svm_tracker.best_expert_idx ~= numel(experts)
                % red rectangle: the prediction of current tracker
                res_prev = svm_tracker.output_exp;
                res_prev(1:2) = res_prev(1:2) - config.padding;
                res_prev = res_prev/config.image_scale;
                rectangle('position',res_prev,'LineWidth',2,'EdgeColor','r') %
                % yellow rectangle: the prediction of the restored tracker
                rectangle('position',res,'LineWidth',2,'EdgeColor','y')   
            else
                % blue rectangle: indicates no restoration happens 
                rectangle('position',res,'LineWidth',2,'EdgeColor','b') 
            end
        end
        
 
        %% update svm classifier
        svm_tracker.temp_count = svm_tracker.temp_count + 1;
        
        if svm_tracker.confidence > config.svm_thresh %&& ~svm_tracker.failure
            train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
            label = sampler.costs(train_mask) < config.thresh_p;
            
            skip_train = false;
            if svm_tracker.confidence > 1.0 
                score_ = -(sampler.patterns_dt(train_mask,:)*svm_tracker.w'+svm_tracker.Bias);
                if prod(double(score_(label) > 1)) == 1 && prod(double(score_(~label)<1)) == 1
                    skip_train = true;
                end
            end
            
            if ~skip_train
                costs = sampler.costs(train_mask);
                fuzzy_weight = ones(size(label));
                fuzzy_weight(~label) = 2*costs(~label)-1;
                updateSvmTracker (sampler.patterns_dt(train_mask,:),label,fuzzy_weight);  
            end
        else % clear update_count
            svm_tracker.update_count = 0;
        end
    end
    
    res = output;
    res(1:2) = res(1:2) - config.padding;
    results.res(frame_id,:) = res/config.image_scale;
end

duration=toc;
results.type='rect';
results.fps=length(seq.s_frames)/duration;
end