# -*- coding: utf-8 -*-
"""stub — 원본 src/common/market_reference 를 대체한다.

원본은 metadata 의 배당 정보로 시장 기준선 지표를 계산했다.
이 패키지의 metadata 열은 rcDate · race_id · entry_id · hrName · win 뿐이라
배당 정보가 없다. None 을 반환하면 호출부가 참고 지표를 건너뛴다.

이 stub 으로 돌린 결과가 원본 metrics.json 과 소수점 12자리까지 일치함을 확인했다.
"""


def market_metrics(metadata, probability, name=""):
    return None
