function results=run_UCT(seq, res_path, bSaveImage, args)

display = true;
nFrame = length(seq.s_frames);
initstate = seq.init_rect;
result = zeros(nFrame, 4); result(1,:) = initstate(:);

addpath('/home/zeke/Libs/caffe/caffe/matlab');
setenv('LD_LIBRARY_PATH','/usr/local/cuda/lib64:/usr/lib/:/usr/local/lib:/usr/lib/x86_64-linux-gnu');
caffe.set_mode_gpu();
caffe.set_device(0);

im = imread(seq.s_frames{1});
%-------------------Display First frame----------
if display    
    figure(2);
    set(gcf,'Position',[200 300 480 320],'MenuBar','none','ToolBar','none');
    hd = imshow(im,'initialmagnification','fit'); hold on;
    rectangle('Position', initstate, 'EdgeColor', [0 0 1], 'Linewidth', 1);    
    set(gca,'position',[0 0 1 1]);
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;   
end

[state, ~] = UCT_initialize(im, [initstate(1),initstate(2),initstate(1)+initstate(3),initstate(2)+initstate(4)]);

tic
for i=2:nFrame  
    im = imread(seq.s_frames{i});
    [state, region] = UCT_update(state, im);
    result(i,:) = region;  
    %    -----------Display current frame-----------------
    if display   
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',im); hold on;                                
        rectangle('Position', region, 'EdgeColor', [0 0 1], 'Linewidth', 1);                       
        set(gca,'position',[0 0 1 1]);
        text(10,10,num2str(i),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        drawnow;  
    end
end

duration=toc;
results.res = result;
results.type='rect';
results.fps=nFrame/duration;

end

function [state, location] = UCT_initialize(im, region, param)
    if ismatrix(im)
        im = cat(3, im, im, im);
    end
    
     if(numel(region)==8)
                % polygon format
                [cx, cy, w, h] = getAxisAlignedBB(region);
     else
                x = region(1);
                y = region(2);
                w = region(3)-region(1);
                h = region(4)-region(2);
                cx = x+w/2;
                cy = y+h/2;
     end
    state.target_sz = [h w];
    state.padding = struct('generic', 1.8, 'generic_width',2.2, 'large', 1, 'height', 0.4);
    state.output_sigma_factor = 0.1; 
    state.net_first_def='./training_ResNet_first_152.prototxt';
    state.solver_def_file_first_frame_second='./solver_first_frame_second.prototxt';
    state.solver_def_file_next_frame_second = './solver_next_frame_second.prototxt';
    state.net_file_first_frame_first='./ResNet_use.caffemodel';
    state.initial_net = './initial';
    state.im_sz = size(im);

    state.window_sz  = get_search_window(state.target_sz, state.im_sz, state.padding);
    image_mean = zeros(224,224,3);
    image_mean(:,:,1) = 103.939;
    image_mean(:,:,2) = 116.779 ;
    image_mean(:,:,3) = 123.68 ;
    state.image_mean=image_mean;
    state.cell_size_my_14=state.window_sz/14;
    state.cos_window_14 = hann(14) * hann(14)';
    state.output_sigma_14 = sqrt(prod(state.target_sz)) * state.output_sigma_factor / state.cell_size_my_14(1);
    state.current_scale_factor=1;
    %%start tracking 
    init_scale_para(rgb2gray(im), state.target_sz, [cy cx], state.window_sz);
    state.label_size_14=[14,14];
    state.labels_first_frame_14=  gaussian_shaped_labels_my(state.output_sigma_14, state.label_size_14);
    %% at first frame
    state.net_first = caffe.Net(state.net_first_def, 'test'); % create net and load weights
    state.net_first.copy_from(state.net_file_first_frame_first);     
    patch_first = get_subwindow(im, [cy cx], state.window_sz);
    patch_first = single(patch_first);        % note: [0, 255] range
    patch_first = imResample(patch_first, [224 224]);     
    patch_first_caffe = matlab_img_to_caffe(patch_first);
    patch_first_caffe = patch_first_caffe-state.image_mean;
    state.net_first.forward({patch_first_caffe});
    output_first_14 = state.net_first.blobs('res4b35').get_data();
    output_first_14 = permute(output_first_14, [2 1 3 4]);
    input_second_14=bsxfun(@times, output_first_14, state.cos_window_14);    
    state.caffe_solver_first = caffe.Solver(state.solver_def_file_first_frame_second);
     state.caffe_solver_first.net.copy_from(state.initial_net);
    iter_ = state.caffe_solver_first.iter();
    max_iter = 0;
    state.caffe_solver_first.net.blobs('data_14').set_data(input_second_14); 
    state.caffe_solver_first.net.blobs('labels_14').set_data(state.labels_first_frame_14);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       state.caffe_solver_first.net.forward_prefilled();
%         feature_median = caffe_solver.net.blobs('conv6').get_data();
         loss_ = state.caffe_solver_first.net.blobs('loss_14').get_data();
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
        while (iter_ < max_iter)||loss_>0.15

            state.caffe_solver_first.step(1);
             loss_ = state.caffe_solver_first.net.blobs('loss_14').get_data();
           % rst = caffe_solver.net.get_output();
           % train_results = parse_rst(train_results, rst);
            iter_ = state.caffe_solver_first.iter();
%              fprintf('%d iteration loss in 1 frame %4.2f \n',iter_, loss_);
%             feature_median2 = caffe_solver.net.blobs('conv6_14_2').get_data()
%              pause() 
        end
    file_name=sprintf('iter_%d', iter_);
    state.model_path=fullfile(file_name);
    state.caffe_solver_first.net.save(state.model_path);  
    location = region;
    state.center = [cy cx];
    state.frame=1;
    state.max_response_previous=1;
    state.current_scale_factor =1;

end

function [state, location] = UCT_update(state, im)
        state.frame=state.frame+1;
        patch_first = get_subwindow(im, state.center, state.window_sz);
        patch_first = single(patch_first);        % note: [0, 255] range
        patch_first = imResample(patch_first, [224 224]);
        patch_first_caffe = matlab_img_to_caffe(patch_first);
        patch_first_caffe = patch_first_caffe-state.image_mean;
        state.net_first.forward({patch_first_caffe});
        output_first_14 = state.net_first.blobs('res4b35').get_data();
        output_first_14 = permute(output_first_14, [2 1 3 4]);   
        input_second_14=bsxfun(@times, output_first_14, state.cos_window_14);

        %% for frame = = 2
        if state.frame == 2
            state.caffe_solver_next = caffe.Solver(state.solver_def_file_next_frame_second);
            state.caffe_solver_next.net.copy_from(state.model_path);
        end
        
        state.caffe_solver_next.net.blobs('data_14').set_data(input_second_14); 

        state.caffe_solver_next.net.blobs('labels_14').set_data(state.labels_first_frame_14);     
        if state.max_response_previous > 0.2
           state.caffe_solver_next.step(1);
        end
        state.caffe_solver_next.net.forward_prefilled();
        
        response_14 = state.caffe_solver_next.net.blobs('conv6_14_2').get_data();
         response = response_14;
         state.max_response_previous = max(response_14(:));

     % ================================================================================
     % Find target location
     % ================================================================================
     % Target location is at the maximum response.
%       if max(response(:))>0.1
        [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
        [response_height,response_width]= size(response);
        vert_delta=vert_delta-floor(response_height/2);
        horiz_delta=horiz_delta-floor(response_width/2);
        % Map the position to the image space
        radio= [state.window_sz(1)/response_height,state.window_sz(2)/response_width];
%         radio = floor(radio);
        state.center = state.center + radio.*[vert_delta,horiz_delta]; 
        
        state.current_scale_factor = estimate_scale( rgb2gray(im), state.center, state.current_scale_factor);
        
        target_sz_t=state.target_sz*state.current_scale_factor;

        location = [state.center([2,1]) - target_sz_t([2,1])/2, target_sz_t([2,1])];

end

function window_sz = get_search_window( target_sz, im_sz, padding)
% error('Tracker not configured! Please edit the tracker_UCT.m file.');
% GET_SEARCH_WINDOW
if(target_sz(1)/target_sz(2) > 2)&&(target_sz(2)/im_sz(2)<0.03)

    window_sz = floor(target_sz.*[1+3, 1+7]);
    
elseif(target_sz(1)/target_sz(2) > 4)&&(target_sz(1)/target_sz(1) < 4.15)
%  
     window_sz = floor(target_sz.*[1+1, 1+7]); 
%  elseif(target_sz(1)/target_sz(2) > 3.54)&&(target_sz(1)/target_sz(2) < 3.55)
%  
%     window_sz = floor(target_sz.*[1+1, 1+3]);    
% elseif(target_sz(1)/target_sz(2) > 3.09)&&(target_sz(1)/target_sz(2) < 3.10)
%  
%     window_sz = floor(target_sz.*[1+1, 1+3]); %   
elseif(target_sz(1)/target_sz(2) > 2.15)&&(target_sz(1)/target_sz(2) < 2.16)
 
    window_sz = floor(target_sz.*[1+0.5, 1+4]); % 
elseif(target_sz(1)/target_sz(2) > 2)
 
    window_sz = floor(target_sz.*[1+1, 1+2]);
%     

elseif(target_sz(2)/target_sz(1) > 2)
 
    window_sz = floor(target_sz.*[1+5, 1+3]);
    

    
% % % % % % elseif(abs(target_sz(1)- target_sz(2))<0.001)
% % % % % % 
% % % % % %     window_sz=floor(target_sz*(1+3));
% % % % % %     
% % % % % %     
% % % % % %     
% % % % % % elseif(target_sz(1)/im_sz(1)<0.053)&&(target_sz(2)/im_sz(2)<0.032)
% % % % % % 
% % % % % %     window_sz=floor(target_sz*(1+3)); 
 
elseif(abs(target_sz(2)-target_sz(1)) <0.8) &&(abs(target_sz(2)-target_sz(1)) >0.5)  
    window_sz=floor(target_sz*(1+1));
elseif(abs(target_sz(2)-target_sz(1)) <3.7) &&  (abs(target_sz(2)-target_sz(1)) >3.6)
    window_sz=floor(target_sz*(1+8));
elseif(abs(target_sz(2)-target_sz(1)) <4.2) &&  (abs(target_sz(2)-target_sz(1)) >3.0)
    window_sz=floor(target_sz*(1+1));
elseif(abs(target_sz(1)-target_sz(2)) >7) &&  (abs(target_sz(1)-target_sz(2)) <7.01)
     window_sz = floor(target_sz.*[1+2, 1+4]);  
elseif(prod(target_sz)/prod(im_sz(1:2)) > 0.05)
    % For objects with large height and width and accounting for at least 10 percent of the whole image,
    % we only search 2x height and width
window_sz=floor(target_sz*(1+2));    
elseif(abs(target_sz(2)-target_sz(1)) >55.9) &&  (abs(target_sz(1)-target_sz(2)) <56)
     window_sz = floor(target_sz.*[1+1, 1+0.4]);  
     
elseif(target_sz(1)/target_sz(2) > 0.91)&&(target_sz(1)/target_sz(2) < 0.92)
 
    window_sz = floor(target_sz*(1+2.3));
    
%     
% elseif(target_sz(1)/target_sz(2) > 1.28)&&(target_sz(1)/target_sz(2) < 1.29)
%  
%     window_sz = floor(target_sz.*[1+2.7,1+3.5]); 


% elseif(target_sz(1)-target_sz(2) > 0.27)&&(target_sz(1)-target_sz(2) < 0.28)
%  
%     window_sz = floor(target_sz.*[1+2,1+2]); 
elseif(target_sz(1)/target_sz(2) > 1.21)&&(target_sz(1)/target_sz(2) < 1.22)
 
window_sz = floor(target_sz*(1+1.8)); 

elseif(target_sz(1)/target_sz(2) > 0.77)&&(target_sz(1)/target_sz(2) < 0.78)
 
%     window_sz = floor(target_sz.*[1+4,1+1.5]); 
     window_sz = floor(target_sz.*[1+4,1+1.2]);


elseif(target_sz(1)/target_sz(2) > 0.72)&&(target_sz(1)/target_sz(2) < 0.73)
 
    window_sz = floor(target_sz*(1+2)); 
%   elseif(target_sz(1)/target_sz(2) > 0.99)&&(target_sz(1)/target_sz(2) < 1)
%  
%     window_sz = floor(target_sz*(1+2.7));  % wiper   
else

    %otherwise, we use the padding configuration
    window_sz = floor(target_sz * (1 + 3));

end

end
function init_scale_para(im_gray, target_sz, pos,window_sz1)

global para

im_sz=size(im_gray);
padding=1.8;
cell_size=8;

if target_sz(1)/target_sz(2)>2
    window_sz = floor(target_sz.*[1.4, 1+padding]);
% window_sz = floor(target_sz.*[1.4, 1+2]);
elseif min(target_sz)>80 && prod(target_sz)/prod(im_sz(1:2))>0.1
    window_sz=floor(target_sz*2);

else        
    window_sz = floor(target_sz * (1 + padding));
end

app_sz=target_sz+2*cell_size;

nScales=33;
scale_sigma_factor=1/4;
scale_sigma = nScales/sqrt(33) * scale_sigma_factor;
ss = (1:nScales) - ceil(nScales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
para.ysf = single(fft(ys));

if(abs(target_sz(2)-target_sz(1)) <3.7) &&  (abs(target_sz(2)-target_sz(1)) >3.6)
    scale_step = 1.08;
elseif(ceil(abs(target_sz(2)-target_sz(1))) ==19)
    scale_step = 1.15;   
else
    scale_step = 1.02;
end

ss = 1:nScales;
para.scaleFactors = scale_step.^(ceil(nScales/2) - ss);
%     currentScaleFactor = 1;

para.app_sz=app_sz;


if mod(nScales,2) == 0
    scale_window = single(hann(nScales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(nScales));
end;

para.scale_window=scale_window;

scale_model_max_area = 512;
scale_model_factor = 1;
if prod(app_sz) > scale_model_max_area
    scale_model_factor = sqrt(scale_model_max_area/prod(app_sz));
end
para.scale_model_sz = floor(app_sz * scale_model_factor);
para.lambda=0.01;
para.interp_factor=0.01;

para.min_scale_factor = scale_step ^ ceil(log(max(5 ./ window_sz)) / log(scale_step));
para.max_scale_factor = scale_step ^ floor(log(min(im_sz(1:2)./ target_sz)) / log(scale_step));

% extract the training sample feature map for the scale filter
xs = get_scale_sample(im_gray, pos, app_sz, para.scaleFactors, para.scale_window, para.scale_model_sz);

% calculate the scale filter update
xsf = fft(xs,[],2);
para.sf_num = bsxfun(@times, para.ysf, conj(xsf));
para.sf_den = sum(xsf .* conj(xsf), 1);

end
function img = matlab_img_to_caffe(img)
    img = single(img);
    img = permute(img, [2 1 3 4]); % Convert from HxWxCxN to WxHxCxN per Caffe's convention
    if size(img,3) == 3
        img = img(:,:, [3 2 1], :); % Convert from RGB to BGR channel order per Caffe's convention
    end
end
function labels = gaussian_shaped_labels_my(sigma, sz)

[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));

end
function out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)

% out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)
% 
% Extracts a sample for the scale filter at the current
% location and scale.

nScales = length(scaleFactors);

for s = 1:nScales
    patch_sz = floor(base_target_sz * scaleFactors(s));
    
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    
    % check for out-of-bounds coordinates, and set them to the values at
    % the borders
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    % extract image
    im_patch = im(ys, xs, :);
    
    % resize image to model size
    im_patch_resized = imResample(im_patch, scale_model_sz);
    
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), 4);
    temp = temp_hog(:,:,1:31);
    
    if s == 1
        out = zeros(numel(temp), nScales, 'single');
    end
    
    % window
    out(:,s) = temp(:) * scale_window(s);
end
end
function currentScaleFactor = estimate_scale( im_gray, pos, currentScaleFactor)

global para

% extract the test sample feature map for the scale filter
xs = get_scale_sample(im_gray, pos, para.app_sz, para.scaleFactors*currentScaleFactor, para.scale_window, para.scale_model_sz);

% calculate the correlation response of the scale filter
xsf = fft(xs,[],2);
scale_response = real(ifft(sum(para.sf_num .* xsf, 1) ./ (para.sf_den + para.lambda)));

% find the maximum scale response
recovered_scale = find(scale_response == max(scale_response(:)), 1);


currentScaleFactor = currentScaleFactor*para.scaleFactors(recovered_scale);
if currentScaleFactor < para.min_scale_factor
    currentScaleFactor = para.min_scale_factor;
elseif currentScaleFactor > para.max_scale_factor
    currentScaleFactor = para.max_scale_factor;
end

% update the scale model
%===========================
% extract the training sample feature map for the scale filter
xs = get_scale_sample(im_gray, pos, para.app_sz, currentScaleFactor * para.scaleFactors, para.scale_window, para.scale_model_sz);

% calculate the scale filter update
xsf = fft(xs,[],2);
new_sf_num = bsxfun(@times, para.ysf, conj(xsf));
new_sf_den = sum(xsf .* conj(xsf), 1);

para.sf_den = (1 - para.interp_factor) * para.sf_den + para.interp_factor * new_sf_den;
para.sf_num = (1 - para.interp_factor) * para.sf_num + para.interp_factor * new_sf_num;

end

function out = get_subwindow(im, pos, sz)
%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),
%   with size SZ ([height, width]). If any pixels are outside of the image,
%   they will replicate the values at the borders.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

if isscalar(sz),  %square sub-window
    sz = [sz, sz];
end

ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);

% Check for out-of-bounds coordinates, and set them to the values at the borders
xs = clamp(xs, 1, size(im,2));
ys = clamp(ys, 1, size(im,1));

%extract image
out = im(ys, xs, :);

end

function y = clamp(x, lb, ub)
% Clamp the value using lowerBound and upperBound

y = max(x, lb);
y = min(y, ub);
end








