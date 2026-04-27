import streamlit as st
import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from tavily import TavilyClient

st.set_page_config(page_title='특금법 챗봇', layout='wide')

st.markdown("""
<style>
.stApp { background-color: #0E1117; }
.main-header { color: #3182ce; font-size: 2rem; font-weight: bold; margin-bottom: 1rem; }
.category-badge-law { background-color: #2b6cb0; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; }
.category-badge-general { background-color: #38a169; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; }
.user-message { padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; background-color: #1a365d; border-left: 4px solid #3182ce; }
.assistant-message { padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; background-color: #1a1a2e; border-left: 4px solid #63b3ed; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header('API 설정')
    openai_key = st.text_input(
        'OpenAI API Key', type='password',
        value=os.environ.get('OPENAI_API_KEY', '')
    )
    tavily_key = st.text_input(
        'Tavily API Key', type='password',
        value=os.environ.get('TAVILY_API_KEY', '')
    )
    if openai_key:
        os.environ['OPENAI_API_KEY'] = openai_key
    if tavily_key:
        os.environ['TAVILY_API_KEY'] = tavily_key
    st.markdown('---')
    st.markdown('**카테고리 안내**')
    st.markdown('<span class="category-badge-law">[특금법]</span> RAG 기반 답변', unsafe_allow_html=True)
    st.markdown('<span class="category-badge-general">[일반]</span> Tavily 웹검색', unsafe_allow_html=True)

TEUKGEUMBEOP_TEXT = (
    '특정 금융거래정보의 보고 및 이용 등에 관한 법률 (약칭: 특금법)'
    '\n[시행 2024. 7. 19.] [법률 제19564호, 2023. 7. 18., 일부개정]'
    '\n\n제1장 총칙'
    '\n\n제1조(목적) 이 법은 금융거래 등을 이용한 자금세탁행위와 공중협박자금조달행위를 '
    '규제하는 데 필요한 특정금융거래정보의 보고 및 이용 등에 관한 사항을 규정함으로써 '
    '범죄행위를 예방하고 나아가 건전하고 투명한 금융거래 질서를 확립하는 데 이바지함을 목적으로 한다.'
    '\n\n제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다.'
    '\n1. 금융회사등이란 금융위원회의 설치 등에 관한 법률 제38조에 따른 검사대상기관, '
    '금융지주회사, 카지노사업자 등을 말한다.'
    '\n2. 금융거래란 금융회사등이 금융자산을 수입 매매 환매 중개 할인 발행 상환 환급 '
    '수탁 등록 교환하거나 그 이자 할인액 또는 배당을 지급하는 것과 이를 대행하는 것을 말한다.'
    '\n3. 자금세탁행위란 범죄수익 등의 취득 또는 처분에 관한 사실을 가장하거나 범죄수익등을 은닉하는 행위를 말한다.'
    '\n4. 공중협박자금조달행위란 테러 목적 및 대량살상무기확산을 위한 자금조달행위를 말한다.'
    '\n5. 가상자산이란 경제적 가치를 지닌 것으로서 전자적으로 거래 또는 이전될 수 있는 전자적 증표를 말한다.'
    '\n6. 가상자산사업자란 가상자산과 관련하여 매도 매수 교환 이전 보관 관리 중개 알선 또는 대행을 영업으로 하는 자를 말한다.'
    '\n\n제3조(의심거래보고 STR) 금융회사등은 불법재산 의심 시 금융정보분석원(FIU)에 보고해야 한다. '
    '보고기한은 3영업일 이내이며 보고사실 누설은 금지된다.'
    '\n\n제4조(금융정보분석원 FIU) 금융위원회 소속으로 STR CTR 정보를 수집 분석하고 '
    '불법거래 의심 시 검찰 경찰 국세청 관세청 등에 정보를 제공한다.'
    '\n\n제4조의2(고액현금거래보고 CTR) 1일 1천만원 이상 현금거래 시 30일 이내에 FIU에 보고해야 한다. '
    '분할거래도 합산하여 보고 대상이 된다.'
    '\n\n제5조(금융회사등의 의무) 고객확인의무 의심거래보고의무 고액현금거래보고의무 등을 이행하여야 한다.'
    '\n\n제5조의2(고객확인의무 CDD) 계좌 신규 개설 일회성 금융거래 의심거래보고 대상 등의 경우 '
    '거래상대방의 신원을 확인해야 한다. 강화된 CDD(EDD)는 고위험 고객(PEP 등), '
    '간소화된 CDD(SDD)는 저위험 고객(정부기관 등)에 적용. 확인 불가 시 거래 거절.'
    '\n\n제5조의3(가상자산사업자 신고) ISMS 인증 + 실명계좌 + 대표자 결격사유 없음의 요건을 갖추어 '
    'FIU에 신고. 미신고 영업 시 5년 이하 징역 또는 5천만원 이하 벌금. 신고수리 후 6개월마다 변경사항 보고.'
    '\n\n제5조의4(가상자산사업자 의무) 고객자산 분리보관 CDD STR '
    '트래블룰(100만원 이상 가상자산 이전 시 송수신인 정보 전달 FATF 권고 16번) '
    '이용자보호 의무. 예치금은 은행에 예치 또는 신탁. 해킹 피해 방지를 위해 보험 가입 또는 준비금 적립.'
    '\n\n제7조(벌칙) 보고사실 누설자 미신고 가상자산사업자: 5년 이하 징역 또는 5천만원 이하 벌금.'
    '\n제8조(과태료) STR CTR 미보고 CDD 미이행: 3천만원 이하 과태료.'
    '\n\nFATF(국제자금세탁방지기구): 1989년 G7에서 설립 40개 권고사항 상호평가 한국 2009년 정회원.'
    '\n자금세탁 3단계: 배치(Placement) 은폐(Layering) 통합(Integration).'
    '\n트래블룰: FATF 권고 16번 근거 100만원 이상 VERIFY 등 솔루션 사용.'
    '\nAML 내부통제: 보고책임자(Compliance Officer) 지정 업무규정 교육훈련 위험평가 감사 거래모니터링.'
)


def setup_rag(api_key):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200,
        separators=['\n\n', '\n', '. ', ' ']
    )
    chunks = splitter.split_text(TEUKGEUMBEOP_TEXT)
    emb = OpenAIEmbeddings(model='text-embedding-3-small', api_key=api_key)
    vs = FAISS.from_texts(texts=chunks, embedding=emb)
    return vs.as_retriever(search_type='similarity', search_kwargs={'k': 4})


def rag_query(question, retriever, llm):
    docs = retriever.invoke(question)
    context = '\n'.join(doc.page_content for doc in docs)
    prompt = PromptTemplate(
        template=(
            '당신은 한국 특금법 전문가입니다.\n'
            '컨텍스트를 기반으로 정확하게 답변하세요.\n'
            '컨텍스트에 없는 내용은 추측하지 마세요.\n\n'
            '컨텍스트:\n{context}\n\n질문: {question}\n\n답변:'
        ),
        input_variables=['context', 'question']
    )
    response = llm.invoke(prompt.format(context=context, question=question))
    return response.content


def tavily_search(query, api_key):
    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=5, include_answer=True)
        answer = resp.get('answer', '')
        results = resp.get('results', [])
        parts = []
        if answer:
            parts.append(f'[Tavily 요약] {answer}')
        for i, r in enumerate(results[:3], 1):
            parts.append(
                f'\n[출처 {i}] {r.get("title", "")}'
                f'\n{r.get("content", "")[:300]}'
                f'\nURL: {r.get("url", "")}'
            )
        return '\n'.join(parts)
    except Exception as e:
        return f'웹 검색 오류: {str(e)}'


def classify_query(question, llm):
    router_prompt = ChatPromptTemplate.from_template(
        '다음 질문을 분류하세요. '
        '카테고리: 특금법(특정금융거래정보법/AML/CDD/STR/CTR/VASP/FIU/트래블룰/FATF) '
        '또는 일반(그 외). 질문: {question} 카테고리(특금법 또는 일반만 출력):'
    )
    chain = router_prompt | llm
    result = chain.invoke({'question': question})
    return '특금법' if '특금법' in result.content.strip() else '일반'


st.markdown('<div class="main-header">특금법 RAG 챗봇</div>', unsafe_allow_html=True)
st.caption('특정 금융거래정보의 보고 및 이용 등에 관한 법률 전문 챗봇')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if msg.get('category'):
            is_law = '특금법' in msg['category']
            cls = 'category-badge-law' if is_law else 'category-badge-general'
            lbl = '[특금법]' if is_law else '[일반]'
            st.markdown(f'<span class="{cls}">{lbl}</span>', unsafe_allow_html=True)
        st.markdown(msg['content'])

if user_input := st.chat_input('질문을 입력하세요...'):
    if not openai_key:
        st.error('사이드바에서 OpenAI API Key를 입력해주세요.')
        st.stop()
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    with st.chat_message('assistant'):
        with st.spinner('답변 생성 중...'):
            try:
                llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=openai_key)
                category = classify_query(user_input, llm)
                if category == '특금법':
                    retriever = setup_rag(openai_key)
                    answer = rag_query(user_input, retriever, llm)
                    cat_label = '특금법 (RAG)'
                else:
                    if not tavily_key:
                        answer = '사이드바에서 Tavily API Key를 입력해주세요.'
                        cat_label = '일반 (키 없음)'
                    else:
                        search_result = tavily_search(user_input, tavily_key)
                        synth = (
                            f'다음 웹 검색 결과를 바탕으로 질문에 한국어로 답변하세요.'
                            f'\n\n질문: {user_input}'
                            f'\n\n검색 결과:\n{search_result}'
                            f'\n\n답변:'
                        )
                        resp = llm.invoke(synth)
                        answer = resp.content
                        cat_label = '일반 (Tavily)'
            except Exception as e:
                answer = f'오류가 발생했습니다: {str(e)}'
                cat_label = '오류'
            is_law = '특금법' in cat_label
            cls = 'category-badge-law' if is_law else 'category-badge-general'
            lbl = '[특금법]' if is_law else '[일반]'
            st.markdown(f'<span class="{cls}">{lbl}</span>', unsafe_allow_html=True)
            st.markdown(answer)
    st.session_state.messages.append(
        {'role': 'assistant', 'content': answer, 'category': cat_label}
    )
