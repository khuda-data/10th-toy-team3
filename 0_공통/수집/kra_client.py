"""
한국마사회 공공데이터포털 API 공통 클라이언트

다양한 KRA Open API(RC경마경주정보, 기수 성적, 경주마 성적 등)를
동일한 인터페이스로 호출할 수 있는 재사용 가능한 클라이언트 클래스.
"""

import logging
import random
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

logger = logging.getLogger(__name__)


class KRAApiError(Exception):
    """KRA API 호출 중 복구 불가능한 에러"""

    def __init__(self, message: str, result_code: str = None, result_msg: str = None):
        super().__init__(message)
        self.result_code = result_code
        self.result_msg = result_msg


class KRAApiClient:
    """한국마사회 공공데이터포털 API 공통 클라이언트

    사용 예시:
        client = KRAApiClient(service_key="YOUR_KEY")
        data = client.fetch_all(
            endpoint="/SeoulRace_1",
            params={"rc_date_fr": "20240101", "rc_date_to": "20240131"}
        )
    """

    BASE_URL = "https://apis.data.go.kr/B551015/API186_1"

    def __init__(
        self,
        service_key: str,
        base_url: str = None,
        num_of_rows: int = 100,
        max_retries: int = 3,
        request_delay: tuple[float, float] = (0.2, 0.3),
    ):
        """
        Args:
            service_key: 공공데이터포털 서비스 인증키 (디코딩된 키)
            base_url: API 기본 URL. 기본값은 RC경마경주정보 API.
            num_of_rows: 페이지당 요청 건수. 기본 100.
            max_retries: 요청 실패 시 최대 재시도 횟수. 기본 3.
            request_delay: 요청 간 딜레이 범위(초). 기본 (0.2, 0.3).
        """
        if not service_key:
            raise ValueError("service_key는 필수입니다. .env 파일을 확인하세요.")

        self.service_key = service_key
        self.base_url = base_url or self.BASE_URL
        self.num_of_rows = num_of_rows
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.session = requests.Session()

    def fetch_all(
        self,
        endpoint: str,
        params: dict = None,
    ) -> list[dict]:
        """전체 페이지를 순회하며 모든 item을 수집하여 반환한다.

        Args:
            endpoint: API 상세기능 경로 (e.g. "/SeoulRace_1")
            params: API별 추가 요청 파라미터 (rc_date_fr, rc_date_to 등)

        Returns:
            수집된 전체 item 리스트 (각 item은 dict)

        Raises:
            KRAApiError: 재시도 초과 또는 복구 불가능한 에러 발생 시
        """
        all_items: list[dict] = []
        page_no = 1
        total_count = None

        while True:
            items, total_count = self._request_page(endpoint, params, page_no)
            all_items.extend(items)

            logger.info(
                f"[페이지 {page_no}] 수집 {len(items)}건 | "
                f"누적 {len(all_items)}건 / 전체 {total_count}건"
            )

            # 종료 조건: 누적 수집건수 >= totalCount
            if total_count == 0 or len(all_items) >= total_count:
                break

            page_no += 1
            # 요청 간 딜레이
            delay = random.uniform(*self.request_delay)
            time.sleep(delay)

        logger.info(f"수집 완료: 총 {len(all_items)}건")
        return all_items

    def _request_page(
        self,
        endpoint: str,
        params: dict | None,
        page_no: int,
    ) -> tuple[list[dict], int]:
        """단일 페이지 요청 + 재시도 로직

        Returns:
            (items, totalCount) 튜플
        """
        request_params = {
            "ServiceKey": self.service_key,
            "pageNo": page_no,
            "numOfRows": self.num_of_rows,
            "_type": "json",
        }
        if params:
            request_params.update(params)

        url = f"{self.base_url}{endpoint}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=request_params, timeout=30)
                response.raise_for_status()

                items, total_count = self._parse_response(response)
                return items, total_count

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else 0
                # 4xx 에러는 재시도 불가 (인증키 오류 등)
                if 400 <= status_code < 500:
                    logger.error(f"HTTP {status_code} 클라이언트 에러 — 재시도 불가: {e}")
                    raise KRAApiError(
                        f"HTTP {status_code} 에러. 서비스키 또는 파라미터를 확인하세요."
                    ) from e
                # 5xx 에러는 재시도
                logger.warning(
                    f"HTTP {status_code} 서버 에러 (시도 {attempt}/{self.max_retries})"
                )

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                logger.warning(
                    f"네트워크 오류 (시도 {attempt}/{self.max_retries}): {e}"
                )

            except KRAApiError:
                # API 결과코드 에러 — 재시도
                if attempt >= self.max_retries:
                    raise
                logger.warning(f"API 응답 에러 (시도 {attempt}/{self.max_retries})")

            # 지수 백오프 대기
            if attempt < self.max_retries:
                wait_time = 2 ** (attempt - 1)  # 1초, 2초, 4초
                logger.info(f"{wait_time}초 후 재시도...")
                time.sleep(wait_time)

        raise KRAApiError(
            f"{self.max_retries}회 재시도 후에도 요청 실패. 수집을 중단합니다."
        )

    def _parse_response(
        self,
        response: requests.Response,
    ) -> tuple[list[dict], int]:
        """응답을 파싱하여 (items, totalCount)를 반환한다.

        JSON 파싱을 우선 시도하고, 실패하면 XML로 전환한다.
        """
        content_type = response.headers.get("Content-Type", "")

        # JSON 시도
        if "json" in content_type or "javascript" in content_type:
            try:
                data = response.json()
                return self._parse_json(data)
            except (ValueError, KeyError):
                logger.warning("JSON 파싱 실패, XML 파싱으로 전환합니다.")
                return self._parse_xml(response.text)

        # XML로 시도
        if "xml" in content_type:
            return self._parse_xml(response.text)

        # Content-Type 불명 — JSON 먼저 시도
        try:
            data = response.json()
            return self._parse_json(data)
        except (ValueError, KeyError):
            return self._parse_xml(response.text)

    def _parse_json(self, data: dict) -> tuple[list[dict], int]:
        """JSON 응답 구조에서 items와 totalCount를 추출한다."""
        header = data.get("response", {}).get("header", {})
        result_code = str(header.get("resultCode", ""))
        result_msg = header.get("resultMsg", "")

        if result_code and result_code != "00":
            raise KRAApiError(
                f"API 에러 — resultCode: {result_code}, resultMsg: {result_msg}",
                result_code=result_code,
                result_msg=result_msg,
            )

        body = data.get("response", {}).get("body", {})
        total_count = int(body.get("totalCount", 0))
        items_data = body.get("items", {})

        # items가 빈 문자열이거나 None인 경우
        if not items_data:
            return [], total_count

        # items > item 추출
        if isinstance(items_data, dict):
            item = items_data.get("item", [])
        else:
            item = items_data

        return self._normalize_items(item), total_count

    def _parse_xml(self, text: str) -> tuple[list[dict], int]:
        """XML 응답 구조에서 items와 totalCount를 추출한다."""
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            raise KRAApiError(f"XML 파싱 실패: {e}") from e

        # 네임스페이스 무시를 위한 태그 검색 헬퍼
        def find_tag(element: ET.Element, tag: str) -> ET.Element | None:
            """하위 요소에서 태그명으로 검색 (네임스페이스 무시)"""
            for child in element.iter():
                local_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if local_tag == tag:
                    return child
            return None

        # resultCode 확인
        header = find_tag(root, "header")
        if header is not None:
            result_code_elem = find_tag(header, "resultCode")
            result_code = result_code_elem.text if result_code_elem is not None else ""
            if result_code and result_code != "00":
                result_msg_elem = find_tag(header, "resultMsg")
                result_msg = result_msg_elem.text if result_msg_elem is not None else ""
                raise KRAApiError(
                    f"API 에러 — resultCode: {result_code}, resultMsg: {result_msg}",
                    result_code=result_code,
                    result_msg=result_msg,
                )

        # totalCount 추출
        total_count_elem = find_tag(root, "totalCount")
        total_count = int(total_count_elem.text) if total_count_elem is not None else 0

        # item 추출
        items: list[dict] = []
        for item_elem in root.iter():
            local_tag = item_elem.tag.split("}")[-1] if "}" in item_elem.tag else item_elem.tag
            if local_tag == "item":
                item_dict = {}
                for child in item_elem:
                    child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    item_dict[child_tag] = child.text
                items.append(item_dict)

        return items, total_count

    def _normalize_items(self, items: Any) -> list[dict]:
        """items가 단일 dict / list / None 등 다양한 형태일 때 list[dict]로 정규화한다."""
        if items is None or items == "" or items == []:
            return []
        if isinstance(items, dict):
            return [items]
        if isinstance(items, list):
            return items
        return []
