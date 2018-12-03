bn = False # Batch normalization

# RPN parameters
anchor_scales = [8, 16, 32]
anchor_ratios = [0.5, 1, 2]
anchor_base_size = 8 # 8x8
rpn_feat_stride = 16 #downsample 16x after vgg16 head (max pool after 4 conv blocks) 
rpn_thresh_low = 0.3
rpn_thresh_high = 0.7
rpn_pre_nms_train = 12000 # num b-boxes to keep before non-maximal suppression
rpn_pos_nms_train = 2000 # num b-boxes to keep after non-maximal suppression
rpn_min_size = 8 # minimum height and width of bbox in original graph

rpn_pre_nms_test = 6000 # num b-boxes to keep before non-maximal suppression during testing
rpn_pos_nms_test = 300 # num b-boxes to keep after non-maximal suppression during testing