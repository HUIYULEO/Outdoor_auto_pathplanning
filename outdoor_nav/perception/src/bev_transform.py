"""Bird's Eye View (BEV) transformation utilities"""
import sys
import os
import cv2 as cv
import numpy as np

from outdoor_nav.utils.utils import img2BEV, test_find_edges, findLowerCorners


class BEVTransformer:
    """Bird's Eye View transformation"""
    
    @staticmethod
    def transform(image):
        """
        Convert front-facing image to BEV
        
        Args:
            image: input image (H, W, 3) or (H, W)
        
        Returns:
            bev_image: bird's eye view image
        """
        return img2BEV(image)
    
    @staticmethod
    def extract_road_edges(bev_mask):
        """
        Extract left and right road edges from BEV mask
        
        Args:
            bev_mask: binary mask in BEV
        
        Returns:
            largest_contour, left_edge, right_edge, bottom_center
        """
        mask, contours, _ = cv.findContours(bev_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = [cv.approxPolyDP(cnt, 2, True) for cnt in contours]
        
        if len(contours) == 0:
            return None, None, None, None
        
        largest_contour = max(contours, key=cv.contourArea)
        lower_left, lower_right = findLowerCorners(largest_contour)
        
        left_edge, right_edge = test_find_edges(bev_mask)
        left_edge.append(lower_left)
        right_edge.append(lower_right)
        
        bottom_center = (
            (lower_left[0] + lower_right[0]) // 2,
            (lower_left[1] + lower_right[1]) // 2
        )
        
        return largest_contour, left_edge, right_edge, bottom_center
