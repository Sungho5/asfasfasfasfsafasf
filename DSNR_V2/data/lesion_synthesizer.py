"""
🔥 DIVERSE LESION SYNTHESIZER: Radiolucent Only with Various Shapes
다양한 형태와 파괴 강도를 가진 투과성 병변만 생성
"""

import cv2
import numpy as np
import random
from scipy.ndimage import distance_transform_edt, gaussian_filter


class DiverseLesionSynthesizer:
    """다양한 형태의 투과성 병변 합성기"""

    def __init__(self):
        self.config = {
            'radiolucent': {
                'lesion_brightness_delta': (-0.30, -0.08),  # 매우 어두움 ~ 약간 어두움
                'size_range': (20, 80),
                'sclerotic_border': True,
                'border_width': (2, 5),
                'border_intensity': (0.05, 0.20),
                'transition_width': (2, 8),  # 부드러움 ~ 날카로움
                'destruction_level': (0.2, 0.9),  # 🔥 약한 파괴 ~ 완전 파괴
                'texture_preservation': (0.0, 0.6),  # 🔥 0.1→0.0: 완전 제거 가능
                'complete_hole_prob': 0.15,  # 🔥 15% 확률로 완전한 구멍
            },
            'mixed': {
                'lesion_brightness_delta': (-0.25, -0.08),
                'size_range': (30, 90),
                'sclerotic_border': True,
                'border_width': (2, 5),
                'border_intensity': (0.05, 0.20),
                'transition_width': (2, 8),
                'destruction_level': (0.3, 0.9),
                'texture_preservation': (0.0, 0.6),  # 🔥 0.1→0.0
                'complete_hole_prob': 0.10,  # 🔥 10% 확률
                'inner_islands': (1, 4),
                'island_size': (10, 35),
                'island_delta': (0.03, 0.18)
            }
        }

    def destroy_bone_structure_extreme(self, image, lesion_mask, config, destruction_mode='variable'):
        """
        🔥 극단적 골 파괴 모드
        - variable: 기존 방식 (다양한 강도)
        - complete_hole: 중심부 완전히 0으로 만들기
        - texture_annihilation: 텍스처 완전 파괴 + 노이즈 패턴
        """
        destroyed = image.copy()

        if lesion_mask.sum() < 5:
            return destroyed

        lesion_coords = np.argwhere(lesion_mask > 0.5)
        lesion_region = image[lesion_mask > 0.5]

        if len(lesion_region) < 10:
            return destroyed

        H, W = image.shape
        lesion_dist = distance_transform_edt(lesion_mask)
        lesion_dist_inv = lesion_dist.max() - lesion_dist
        lesion_dist_inv = lesion_dist_inv / (lesion_dist_inv.max() + 1e-6)

        if destruction_mode == 'complete_hole':
            # 🔥 완전한 구멍: 중심부는 0, 가장자리는 점진적으로
            for y, x in lesion_coords:
                center_factor = lesion_dist_inv[y, x]  # 0(가장자리) ~ 1(중심)

                if center_factor > 0.7:
                    # 중심부: 완전히 0
                    destroyed[y, x] = 0.0
                elif center_factor > 0.4:
                    # 중간부: 매우 어둡게
                    destroyed[y, x] = image[y, x] * 0.1
                else:
                    # 가장자리: 점진적으로 감소
                    fade_factor = center_factor / 0.4
                    destroyed[y, x] = image[y, x] * (1 - fade_factor * 0.9)

        elif destruction_mode == 'texture_annihilation':
            # 🔥 텍스처 완전 파괴: 원본 texture 제거 + 파괴된 노이즈 패턴
            # 1. 매우 강한 blur로 texture 제거
            very_blurred = gaussian_filter(image, sigma=5.0)

            # 2. 파괴된 골 구조를 시뮬레이션하는 노이즈 패턴
            noise_pattern = np.random.randn(H, W) * 0.08
            noise_pattern = gaussian_filter(noise_pattern, sigma=1.2)

            # 3. 밝기 크게 감소
            destruction_level = random.uniform(0.7, 0.9)
            texture_preservation = random.uniform(0.0, 0.1)  # 거의 보존 안함

            delta_min, delta_max = config['lesion_brightness_delta']
            delta = random.uniform(delta_min * 1.5, delta_max * 0.5)  # 더 어둡게

            for y, x in lesion_coords:
                center_factor = lesion_dist_inv[y, x]
                local_destruction = destruction_level * (0.5 + 0.5 * center_factor)

                original = image[y, x]
                base = original + delta
                noise = noise_pattern[y, x]
                blurred = very_blurred[y, x]

                # 거의 모든 원본 texture 제거, 파괴된 노이즈로 대체
                destroyed[y, x] = (
                    blurred * 0.3 +  # 약간의 blurred 원본
                    base * local_destruction +
                    original * (1 - local_destruction) * texture_preservation +
                    noise * 0.7  # 강한 노이즈
                )

        else:  # variable (기존 방식)
            destruction_level = random.uniform(*config['destruction_level'])
            texture_preservation = random.uniform(*config['texture_preservation'])

            delta_min, delta_max = config['lesion_brightness_delta']
            delta = random.uniform(delta_min, delta_max)

            noise_low = np.random.randn(H, W) * 0.015
            noise_low = gaussian_filter(noise_low, sigma=2.5)

            noise_high = np.random.randn(H, W) * 0.008
            noise_high = gaussian_filter(noise_high, sigma=0.3)

            noise_combined = noise_low + noise_high

            for y, x in lesion_coords:
                center_factor = lesion_dist_inv[y, x]
                local_destruction = destruction_level * (0.3 + 0.7 * center_factor)

                original = image[y, x]
                base = original + delta
                noise = noise_combined[y, x]

                destroyed[y, x] = (
                    base * local_destruction +
                    original * (1 - local_destruction) * texture_preservation +
                    noise
                )

        return np.clip(destroyed, 0, 1)

    def add_sclerotic_border(self, image, lesion_mask, border_width=3, border_intensity=0.1):
        """경화성 경계"""
        result = image.copy()

        if lesion_mask.sum() < 5:
            return result

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border_width + 1, 2 * border_width + 1))
        dilated = cv2.dilate(lesion_mask.astype(np.uint8), kernel, iterations=1)
        border_ring = (dilated - lesion_mask.astype(np.uint8)).astype(np.float32)
        border_soft = gaussian_filter(border_ring, sigma=1.5)

        result = result + border_soft * border_intensity
        return np.clip(result, 0, 1)

    def create_irregular_boundary(self, lesion_mask, irregularity=0.2):
        """불규칙한 경계"""
        if lesion_mask.sum() < 5:
            return lesion_mask

        noise = np.random.randn(*lesion_mask.shape) * irregularity
        noise_filtered = gaussian_filter(noise, sigma=2.5)
        distorted = lesion_mask + noise_filtered
        distorted = gaussian_filter(distorted, sigma=2.0)

        threshold = 0.5 - irregularity * 0.1
        irregular_mask = (distorted > threshold).astype(np.float32)

        return irregular_mask

    def create_soft_mask(self, mask, transition_width=4):
        """부드러운 경계"""
        sigma = max(0.8, transition_width * 0.6)
        soft_mask = gaussian_filter(mask.astype(np.float32), sigma=sigma)
        return np.clip(soft_mask, 0, 1)

    def create_shape_circle(self, lesion_mask, cx, cy, size):
        """원형"""
        cv2.circle(lesion_mask, (cx, cy), size // 2, 1, -1)
        return lesion_mask

    def create_shape_ellipse(self, lesion_mask, cx, cy, size):
        """타원형"""
        aspect = random.uniform(0.5, 2.0)
        angle = random.randint(0, 180)
        cv2.ellipse(lesion_mask, (cx, cy),
                   (int(size / 2), int(size * aspect / 2)),
                   angle, 0, 360, 1, -1)
        return lesion_mask

    def create_shape_irregular_circle(self, lesion_mask, cx, cy, size):
        """불규칙한 원형 (노이즈가 많은 원)"""
        cv2.circle(lesion_mask, (cx, cy), size // 2, 1, -1)

        irregularity = random.uniform(0.2, 0.4)
        noise = np.random.randn(*lesion_mask.shape) * irregularity
        noise_filtered = gaussian_filter(noise, sigma=3.0)
        distorted = lesion_mask + noise_filtered
        distorted = gaussian_filter(distorted, sigma=1.5)

        irregular_mask = (distorted > 0.4).astype(np.float32)
        return irregular_mask

    def create_shape_irregular_ellipse(self, lesion_mask, cx, cy, size):
        """불규칙한 타원형"""
        aspect = random.uniform(0.5, 2.5)
        angle = random.randint(0, 180)
        cv2.ellipse(lesion_mask, (cx, cy),
                   (int(size / 2), int(size * aspect / 2)),
                   angle, 0, 360, 1, -1)

        irregularity = random.uniform(0.15, 0.35)
        noise = np.random.randn(*lesion_mask.shape) * irregularity
        noise_filtered = gaussian_filter(noise, sigma=2.8)
        distorted = lesion_mask + noise_filtered
        distorted = gaussian_filter(distorted, sigma=1.8)

        irregular_mask = (distorted > 0.45).astype(np.float32)
        return irregular_mask

    def create_shape_teardrop(self, lesion_mask, cx, cy, size):
        """눈물 모양 (물방울)"""
        main_radius = int(size * 0.4)
        cv2.circle(lesion_mask, (cx, cy), main_radius, 1, -1)

        angle = random.uniform(0, 2 * np.pi)
        tail_length = int(size * 0.5)
        num_tail_circles = 5

        for i in range(num_tail_circles):
            t = (i + 1) / num_tail_circles
            offset = tail_length * t
            tail_radius = int(main_radius * (1 - t * 0.8))

            tail_x = int(cx + offset * np.cos(angle))
            tail_y = int(cy + offset * np.sin(angle))

            cv2.circle(lesion_mask, (tail_x, tail_y), tail_radius, 1, -1)

        lesion_mask = gaussian_filter(lesion_mask, sigma=1.5)
        lesion_mask = (lesion_mask > 0.3).astype(np.float32)

        return lesion_mask

    def create_shape_grape_cluster(self, lesion_mask, cx, cy, size):
        """포도송이 모양 (여러 원들이 뭉쳐있는 형태)"""
        main_radius = int(size * 0.35)
        cv2.circle(lesion_mask, (cx, cy), main_radius, 1, -1)

        num_grapes = random.randint(4, 7)

        for i in range(num_grapes):
            angle = (2 * np.pi * i / num_grapes) + random.uniform(-0.3, 0.3)
            distance = main_radius * random.uniform(0.6, 0.9)
            grape_radius = int(main_radius * random.uniform(0.4, 0.7))

            grape_x = int(cx + distance * np.cos(angle))
            grape_y = int(cy + distance * np.sin(angle))

            cv2.circle(lesion_mask, (grape_x, grape_y), grape_radius, 1, -1)

        lesion_mask = gaussian_filter(lesion_mask, sigma=2.0)
        lesion_mask = (lesion_mask > 0.25).astype(np.float32)

        return lesion_mask

    def create_shape_multilocular_distinct(self, lesion_mask, cx, cy, size):
        """
        다방성 병변 - 경계가 살아있는 형태
        여러 원/타원이 겹쳐있지만 각각의 경계선이 명확하게 보임
        """
        H, W = lesion_mask.shape
        num_locules = random.randint(3, 6)  # 3~6개의 작은 낭종

        # 각 locule을 개별적으로 저장
        individual_masks = []

        # 첫 번째 메인 locule
        main_radius = int(size * 0.35)
        main_mask = np.zeros_like(lesion_mask)
        cv2.circle(main_mask, (cx, cy), main_radius, 1, -1)
        individual_masks.append(main_mask)

        # 추가 locules (메인 주변에 겹치게 배치)
        for i in range(num_locules - 1):
            angle = random.uniform(0, 2 * np.pi)
            # 겹치게 하기 위해 거리를 줄임
            distance = main_radius * random.uniform(0.4, 0.7)  # 겹침 정도

            locule_cx = int(cx + distance * np.cos(angle))
            locule_cy = int(cy + distance * np.sin(angle))

            # 크기를 다양하게
            if random.random() > 0.5:
                # 원형
                locule_radius = int(main_radius * random.uniform(0.5, 0.8))
                locule_mask = np.zeros_like(lesion_mask)
                cv2.circle(locule_mask, (locule_cx, locule_cy), locule_radius, 1, -1)
            else:
                # 타원형
                aspect = random.uniform(0.6, 1.5)
                angle_deg = random.randint(0, 180)
                locule_radius = int(main_radius * random.uniform(0.5, 0.8))
                locule_mask = np.zeros_like(lesion_mask)
                cv2.ellipse(locule_mask, (locule_cx, locule_cy),
                           (locule_radius, int(locule_radius * aspect)),
                           angle_deg, 0, 360, 1, -1)

            individual_masks.append(locule_mask)

        # 🔥 핵심: 각 locule의 경계를 살리면서 합치기
        # 방법: 각 locule에 서로 다른 강도를 주고, max로 합치면 경계가 보임

        combined_mask = np.zeros_like(lesion_mask)

        for i, mask in enumerate(individual_masks):
            # 각 locule을 약간씩 blur (경계는 살리면서 자연스럽게)
            soft_mask = gaussian_filter(mask.astype(np.float32), sigma=0.8)

            # 경계 강조를 위해 각 locule마다 약간씩 다른 값 부여
            # 겹친 부분에서 경계가 보이도록
            intensity = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, 0.9 교대로

            combined_mask = np.maximum(combined_mask, soft_mask * intensity)

        # 최종 마스크 정규화
        if combined_mask.max() > 0:
            combined_mask = combined_mask / combined_mask.max()

        # 약간의 threshold로 경계를 더 명확하게
        combined_mask = (combined_mask > 0.3).astype(np.float32)

        # 매우 약한 blur만 적용 (경계는 유지)
        combined_mask = gaussian_filter(combined_mask, sigma=0.5)

        return combined_mask

    def synthesize_radiolucent_diverse(self, image, lesion_mask, lesion_info):
        """다양한 형태의 투과성 병변"""
        config = self.config['radiolucent']

        if lesion_mask.sum() == 0:
            return image, None

        cx, cy, size = lesion_info

        irregularity = random.uniform(0.05, 0.15)
        lesion_mask = self.create_irregular_boundary(lesion_mask, irregularity)

        margin = size + 25
        y1, y2 = max(0, cy - margin), min(image.shape[0], cy + margin)
        x1, x2 = max(0, cx - margin), min(image.shape[1], cx + margin)

        local_image = image[y1:y2, x1:x2].copy()
        local_mask = lesion_mask[y1:y2, x1:x2]

        # 🔥 파괴 모드 선택
        rand = random.random()
        if rand < config['complete_hole_prob']:
            destruction_mode = 'complete_hole'
        elif rand < config['complete_hole_prob'] + 0.15:  # 추가 15% 확률
            destruction_mode = 'texture_annihilation'
        else:
            destruction_mode = 'variable'

        local_destroyed = self.destroy_bone_structure_extreme(
            local_image, local_mask, config, destruction_mode
        )

        # Sclerotic border (완전한 구멍일 때는 경계 없음)
        if config['sclerotic_border'] and destruction_mode != 'complete_hole' and random.random() > 0.4:
            border_width = random.randint(*config['border_width'])
            border_intensity = random.uniform(*config['border_intensity'])
            local_destroyed = self.add_sclerotic_border(
                local_destroyed, local_mask, border_width, border_intensity
            )

        transition_width = random.randint(*config['transition_width'])
        soft_mask = self.create_soft_mask(local_mask, transition_width)
        local_blended = soft_mask * local_destroyed + (1 - soft_mask) * local_image

        blur_strength = random.choice([0, 1, 3])
        if blur_strength > 0:
            local_final = cv2.GaussianBlur(local_blended, (blur_strength, blur_strength), 0.5)
        else:
            local_final = local_blended

        result = image.copy()
        result[y1:y2, x1:x2] = local_final

        lesion_region_before = image[lesion_mask > 0.5]
        lesion_region_after = result[lesion_mask > 0.5]

        if len(lesion_region_before) > 0 and len(lesion_region_after) > 0:
            avg_delta = np.mean(lesion_region_after) - np.mean(lesion_region_before)
        else:
            avg_delta = 0.0

        return np.clip(result, 0, 1), avg_delta

    def synthesize_mixed_diverse(self, image, lesion_mask, lesion_info):
        """혼합형 병변 (어두운 배경 + 작은 밝은 섬들)"""
        config = self.config['mixed']

        x0, delta_outer = self.synthesize_radiolucent_diverse(image, lesion_mask, lesion_info)

        n_islands = random.randint(*config['inner_islands'])
        lesion_coords = np.argwhere(lesion_mask > 0.5)

        if len(lesion_coords) > 10:
            for _ in range(n_islands):
                idx = random.randint(0, len(lesion_coords) - 1)
                cy, cx = lesion_coords[idx]
                island_size = random.randint(*config['island_size'])

                island_mask = np.zeros_like(lesion_mask)
                cv2.circle(island_mask, (cx, cy), island_size // 2, 1, -1)
                island_mask = island_mask * lesion_mask

                island_delta = random.uniform(*config['island_delta'])
                island_soft = self.create_soft_mask(island_mask, transition_width=2)
                x0 = x0 + island_soft * island_delta

        return np.clip(x0, 0, 1), delta_outer

    def create_lesion_mask(self, roi_mask, lesion_type='radiolucent'):
        """
        🔥 다양한 형태의 병변 마스크 생성
        - circle: 원형
        - ellipse: 타원형
        - irregular_circle: 불규칙한 원형
        - irregular_ellipse: 불규칙한 타원형
        - teardrop: 눈물 방울 모양
        - grape_cluster: 포도송이 (경계 흐림)
        - multilocular_distinct: 다방성 (경계 선명) 🔥NEW
        """
        H, W = roi_mask.shape
        dist_map = distance_transform_edt(roi_mask)
        dist_map = dist_map / (dist_map.max() + 1e-6)

        prob_map = np.power(dist_map, 0.5)
        prob_map[roi_mask == 0] = 0

        if prob_map.sum() == 0:
            return np.zeros_like(roi_mask), None

        config = self.config[lesion_type]
        size_min, size_max = config['size_range']

        shape_choices = [
            'circle',
            'ellipse',
            'irregular_circle',
            'irregular_ellipse',
            'teardrop',
            'grape_cluster',
            'multilocular_distinct'  # 🔥 경계가 살아있는 다방성
        ]

        for attempts in range(50):
            prob_flat = prob_map.flatten()
            prob_flat = prob_flat / (prob_flat.sum() + 1e-8)
            center_idx = np.random.choice(len(prob_flat), p=prob_flat)
            cy, cx = divmod(center_idx, W)

            size = random.randint(size_min, size_max)
            lesion_mask = np.zeros_like(roi_mask, dtype=np.float32)

            shape = random.choice(shape_choices)

            if shape == 'circle':
                lesion_mask = self.create_shape_circle(lesion_mask, cx, cy, size)
            elif shape == 'ellipse':
                lesion_mask = self.create_shape_ellipse(lesion_mask, cx, cy, size)
            elif shape == 'irregular_circle':
                lesion_mask = self.create_shape_irregular_circle(lesion_mask, cx, cy, size)
            elif shape == 'irregular_ellipse':
                lesion_mask = self.create_shape_irregular_ellipse(lesion_mask, cx, cy, size)
            elif shape == 'teardrop':
                lesion_mask = self.create_shape_teardrop(lesion_mask, cx, cy, size)
            elif shape == 'grape_cluster':
                lesion_mask = self.create_shape_grape_cluster(lesion_mask, cx, cy, size)
            elif shape == 'multilocular_distinct':
                lesion_mask = self.create_shape_multilocular_distinct(lesion_mask, cx, cy, size)

            lesion_mask = lesion_mask * roi_mask

            if lesion_mask.sum() > 50:
                return lesion_mask, (cx, cy, size)

        return np.zeros_like(roi_mask), None

    def synthesize(self, image, roi_mask, lesion_type='random'):
        """
        🔥 다양한 병변 합성 (2~4개, radiolucent와 mixed만)
        """
        num_lesions = random.randint(2, 4)

        combined_lesion_mask = np.zeros_like(roi_mask)
        x0 = image.copy()
        deltas = []
        lesion_types_list = []

        for i in range(num_lesions):
            if lesion_type == 'random':
                current_type = random.choice(['radiolucent', 'radiolucent', 'mixed'])
            else:
                current_type = lesion_type

            for attempts in range(30):
                lesion_mask, lesion_info = self.create_lesion_mask(roi_mask, current_type)

                if lesion_mask.sum() == 0 or lesion_info is None:
                    continue

                if combined_lesion_mask.sum() > 0:
                    intersection = (combined_lesion_mask * lesion_mask).sum()
                    union = combined_lesion_mask.sum() + lesion_mask.sum() - intersection
                    iou = intersection / (union + 1e-8)
                    if iou > 0.2:
                        continue

                if current_type == 'radiolucent':
                    x0, delta = self.synthesize_radiolucent_diverse(x0, lesion_mask, lesion_info)
                else:
                    x0, delta = self.synthesize_mixed_diverse(x0, lesion_mask, lesion_info)

                if delta is not None:
                    deltas.append(delta)
                    lesion_types_list.append(current_type)
                    combined_lesion_mask = np.clip(combined_lesion_mask + lesion_mask, 0, 1)

                break

        x0 = x0 * roi_mask + image * (1 - roi_mask)

        avg_delta = np.mean(deltas) if len(deltas) > 0 else 0.0
        lesion_types_str = f"{len(deltas)} lesions: {', '.join(set(lesion_types_list))}"

        return x0, combined_lesion_mask, lesion_types_str, avg_delta


# 호환성 별칭
StrongLesionSynthesizer = DiverseLesionSynthesizer
AnatomicalLesionSynthesizer = DiverseLesionSynthesizer
