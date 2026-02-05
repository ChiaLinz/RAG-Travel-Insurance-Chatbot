"""
Main - 旅遊保險問答系統主程序（優化版）

✅ 新增功能：
1. 預載模式（單例模式，避免重複初始化）
2. 批量查詢模式
3. 修復相似度顯示問題
4. 更完善的錯誤處理

使用流程：
1. python main.py build - 構建索引
2. python main.py query "your question" - 單次查詢
3. python main.py interactive - 互動式問答（預載引擎）
4. python main.py batch questions.txt - 批量查詢
5. python main.py stats - 查看統計
"""

import argparse
import os
import sys
from typing import Optional
from config import INDEX_DIR
from core.chunk_generator import generate_chunks_from_pdf, save_chunks
from core.intent_generator import generate_all_intents, save_intents, save_chunks_with_intents
from core.retrieval_engine import RetrievalEngine
from core.answer_generator import AnswerGenerator, ConversationalAnswerGenerator
import json


# ==================== 全局變量（預載模式）====================

_retrieval_engine: Optional[RetrievalEngine] = None
_answer_generator: Optional[AnswerGenerator] = None


def get_engine():
    """
    獲取預載的引擎實例（單例模式）
    
    優點：
    - 避免重複初始化（節省時間）
    - 內存中保留 embeddings（提升性能）
    - 適合互動式和批量查詢
    
    Returns:
        (RetrievalEngine, AnswerGenerator) 元組
    """
    global _retrieval_engine, _answer_generator
    
    if _retrieval_engine is None or _answer_generator is None:
        intents_path = os.path.join(INDEX_DIR, "intents.json")
        chunks_path = os.path.join(INDEX_DIR, "chunks_structured_with_intents.json")
        
        if not os.path.exists(intents_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(
                "❌ 索引不存在，請先運行: python main.py build"
            )
        
        print("🔄 初始化檢索引擎...")
        _retrieval_engine = RetrievalEngine(
            intents_path, 
            chunks_path,
            use_bm25=True,
            use_cross_encoder=True
        )
        _answer_generator = AnswerGenerator(_retrieval_engine)
        print("✅ 引擎初始化完成\n")
    
    return _retrieval_engine, _answer_generator


# ==================== 構建索引 ====================

def build_index(regenerate: bool = False):
    """
    構建索引（chunks 和 intents）
    
    Args:
        regenerate: 是否強制重新生成
    """
    chunks_path = os.path.join(INDEX_DIR, "chunks_structured.json")
    intents_path = os.path.join(INDEX_DIR, "intents.json")
    chunks_with_intents_path = os.path.join(INDEX_DIR, "chunks_structured_with_intents.json")
    
    # 檢查是否已存在
    if not regenerate and os.path.exists(chunks_with_intents_path) and os.path.exists(intents_path):
        print("✅ 索引已存在。使用 --regenerate 強制重新生成。")
        return
    
    print("\n" + "="*80)
    print("🔨 開始構建索引")
    print("="*80)
    
    # Step 1: 生成 chunks
    print("\n【Step 1/3】生成條文 Chunks...")
    print("-" * 80)
    chunks = generate_chunks_from_pdf()
    save_chunks(chunks, chunks_path)
    
    # 轉換為字典格式
    chunks_dict = [chunk.to_dict() for chunk in chunks]
    
    # Step 2: 生成 intents
    print("\n【Step 2/3】生成意圖索引...")
    print("-" * 80)
    intents = generate_all_intents(
        chunks_dict, 
        generate_for_items=True,
        generate_comparisons=True
    )
    save_intents(intents, intents_path)
    
    # Step 3: 保存帶意圖的 chunks
    print("\n【Step 3/3】保存完整索引...")
    print("-" * 80)
    save_chunks_with_intents(chunks_dict, chunks_with_intents_path)
    
    print("\n" + "="*80)
    print("✅ 索引構建完成！")
    print("="*80)
    print(f"📁 輸出文件:")
    print(f"   - {chunks_path}")
    print(f"   - {intents_path}")
    print(f"   - {chunks_with_intents_path}")


# ==================== 單次查詢 ====================

def single_query(query: str, 
                top_k_intents: int = 5,
                top_k_clauses: int = 3,
                show_sources: bool = True,
                show_details: bool = False):
    """
    單次查詢
    
    Args:
        query: 查詢問題
        top_k_intents: 檢索 top-k 意圖
        top_k_clauses: 返回 top-k 條文
        show_sources: 是否顯示來源
        show_details: 是否顯示詳細信息（intent 分數等）
    """
    try:
        # 使用預載引擎
        _, answer_gen = get_engine()
        
        # 生成答案
        print(f"📝 問題: {query}")
        print("=" * 80)
        
        result = answer_gen.generate(
            query,
            top_k_intents=top_k_intents,
            top_k_clauses=top_k_clauses,
            include_sources=show_sources
        )
        
        # 顯示答案
        print(f"\n💬 答案:\n{result['answer']}")
        
        # 顯示來源
        if show_sources and 'sources' in result:
            print("\n" + "-" * 80)
            print("📚 參考條文:")
            for source in result['sources']:
                location = source['clause_id']
                if source.get('item_no'):
                    location += f" 第{source['item_no']}項"
                if source.get('subitem_no'):
                    location += f" ({source['subitem_no']})"
                # ✅ 現在應該顯示正確的相似度
                print(f"  - {location} (相似度: {source['similarity_score']:.3f})")
        
        # 顯示詳細信息
        if show_details and 'top_intents' in result:
            print("\n" + "-" * 80)
            print("🎯 Top Intents:")
            for i, intent in enumerate(result['top_intents'][:3], 1):
                print(f"  {i}. {intent['user_query']}")
                print(f"     分數: {intent.get('similarity', 0):.3f}")
        
        # 顯示 token 使用
        if 'usage' in result:
            print("\n" + "-" * 80)
            print(f"💡 Token 使用: {result['usage']['total_tokens']} tokens "
                  f"(prompt: {result['usage']['prompt_tokens']}, "
                  f"completion: {result['usage']['completion_tokens']})")
        
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


# ==================== 互動式問答（優化版）====================

def interactive_mode():
    """互動式問答模式（預載引擎，提升性能）"""
    try:
        # 預載引擎
        retrieval_engine, _ = get_engine()
        answer_gen = ConversationalAnswerGenerator(retrieval_engine)
        
        print("="*80)
        print("🤖 旅遊保險問答系統 - 互動模式")
        print("="*80)
        print("💡 提示:")
        print("   - 輸入問題並按 Enter")
        print("   - 輸入 'clear' 清除對話歷史")
        print("   - 輸入 'quit' 或 'exit' 退出")
        print("="*80)
        
        while True:
            try:
                # 獲取用戶輸入
                query = input("\n📝 您的問題: ").strip()
                
                if not query:
                    continue
                
                # 特殊命令
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 再見！")
                    break
                
                if query.lower() == 'clear':
                    answer_gen.reset_history()
                    print("✅ 對話歷史已清除")
                    continue
                
                # 生成答案
                print("-" * 80)
                result = answer_gen.generate_with_history(
                    query,
                    top_k_intents=5,
                    top_k_clauses=3
                )
                
                print(f"\n💬 答案:\n{result['answer']}")
                
                # 顯示來源
                if 'sources' in result:
                    print("\n📚 參考條文:")
                    for source in result['sources']:
                        location = source['clause_id']
                        if source.get('item_no'):
                            location += f" 第{source['item_no']}項"
                        print(f"  - {location} (相似度: {source['similarity_score']:.3f})")
                
                if 'conversation_length' in result:
                    print(f"\n💡 對話輪數: {result['conversation_length']}")
                
                if 'usage' in result:
                    print(f"💡 Token: {result['usage']['total_tokens']}")
                
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n\n👋 再見！")
                break
            except Exception as e:
                print(f"\n❌ 錯誤: {e}")
    
    except FileNotFoundError as e:
        print(str(e))


# ==================== 批量查詢 ====================

def batch_query(input_file: str, output_file: Optional[str] = None):
    """
    批量查詢模式（新增功能）
    
    Args:
        input_file: 輸入文件（每行一個問題）
        output_file: 輸出文件（可選，JSON格式）
    """
    try:
        # 預載引擎（避免每次查詢都初始化）
        _, answer_gen = get_engine()
        
        # 讀取問題
        with open(input_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
        
        print(f"\n📋 批量查詢: {len(questions)} 個問題")
        print("=" * 80)
        
        results = []
        
        for i, query in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {query}")
            print("-" * 80)
            
            result = answer_gen.generate(
                query,
                top_k_intents=5,
                top_k_clauses=3,
                include_sources=True
            )
            
            print(f"\n{result['answer']}")
            
            if 'sources' in result:
                print("\n📚 參考條文:")
                for source in result['sources']:
                    location = source['clause_id']
                    print(f"  - {location} (相似度: {source['similarity_score']:.3f})")
            
            results.append({
                "question": query,
                "answer": result['answer'],
                "sources": [
                    {
                        "clause_id": s['clause_id'],
                        "score": s['similarity_score']
                    }
                    for s in result.get('sources', [])
                ],
                "token_usage": result.get('usage', {})
            })
            
            print("=" * 80)
        
        # 保存結果
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 結果已保存至: {output_file}")
        
        # 統計
        total_tokens = sum(r.get('token_usage', {}).get('total_tokens', 0) for r in results)
        print(f"\n📊 總計:")
        print(f"   - 問題數: {len(results)}")
        print(f"   - Token 使用: {total_tokens}")
        print(f"   - 平均 Token: {total_tokens / len(results):.0f}")
    
    except FileNotFoundError as e:
        if "intents.json" in str(e) or "chunks" in str(e):
            print(str(e))
        else:
            print(f"❌ 找不到輸入文件: {input_file}")
    except Exception as e:
        print(f"❌ 錯誤: {e}")


# ==================== 查看統計信息 ====================

def show_stats():
    """顯示索引統計信息"""
    chunks_path = os.path.join(INDEX_DIR, "chunks_structured_with_intents.json")
    intents_path = os.path.join(INDEX_DIR, "intents.json")
    
    if not os.path.exists(chunks_path) or not os.path.exists(intents_path):
        print("❌ 索引不存在，請先運行: python main.py build")
        return
    
    # 載入數據
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    with open(intents_path, "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    
    chunks = chunks_data["chunks"]
    intents = intents_data["intents"]
    
    # 統計
    total_items = sum(len(chunk["clause"]["items"]) for chunk in chunks)
    total_subitems = sum(
        len(item["sub_items"])
        for chunk in chunks
        for item in chunk["clause"]["items"]
    )
    
    # 意圖分類統計
    categories = {}
    query_types = {}
    for intent in intents:
        cat = intent.get("category", "未分類")
        categories[cat] = categories.get(cat, 0) + 1
        
        qt = intent.get("query_type", "其他")
        query_types[qt] = query_types.get(qt, 0) + 1
    
    # 顯示
    print("\n" + "="*80)
    print("📊 索引統計信息")
    print("="*80)
    
    print("\n【條文結構】")
    print(f"  總條文數: {len(chunks)}")
    print(f"  總項目數: {total_items}")
    print(f"  總款項數: {total_subitems}")
    
    print("\n【意圖索引】")
    print(f"  總意圖數: {len(intents)}")
    print(f"  平均每條文意圖數: {len(intents) / len(chunks):.1f}")
    
    print("\n【意圖類型】")
    for qt, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(intents) * 100
        print(f"  {qt}: {count} ({percentage:.1f}%)")
    
    print("\n【意圖分類 (Top 10)】")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = count / len(intents) * 100
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    print("\n【索引文件】")
    chunks_size = os.path.getsize(chunks_path) / 1024
    intents_size = os.path.getsize(intents_path) / 1024
    print(f"  chunks 文件: {chunks_size:.1f} KB")
    print(f"  intents 文件: {intents_size:.1f} KB")
    
    print("="*80)


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description="旅遊保險問答系統（優化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py build                          # 構建索引
  python main.py build --regenerate             # 強制重新構建
  python main.py query "什麼情況下可以理賠？"    # 單次查詢
  python main.py query "..." --details          # 顯示詳細信息
  python main.py interactive                    # 互動模式（預載引擎）
  python main.py batch questions.txt            # 批量查詢
  python main.py batch questions.txt -o out.json  # 批量查詢並保存
  python main.py stats                          # 查看統計
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # build 命令
    build_parser = subparsers.add_parser('build', help='構建索引')
    build_parser.add_argument('--regenerate', action='store_true',
                            help='強制重新生成索引')
    
    # query 命令
    query_parser = subparsers.add_parser('query', help='單次查詢')
    query_parser.add_argument('question', type=str, help='要查詢的問題')
    query_parser.add_argument('--top-k-intents', type=int, default=5,
                            help='檢索 top-k 意圖 (默認: 5)')
    query_parser.add_argument('--top-k-clauses', type=int, default=3,
                            help='返回 top-k 條文 (默認: 3)')
    query_parser.add_argument('--no-sources', action='store_true',
                            help='不顯示來源信息')
    query_parser.add_argument('--details', action='store_true',
                            help='顯示詳細信息（intent 分數等）')
    
    # interactive 命令
    subparsers.add_parser('interactive', help='互動式問答模式（預載引擎）')
    
    # batch 命令
    batch_parser = subparsers.add_parser('batch', help='批量查詢')
    batch_parser.add_argument('input', type=str, help='輸入文件（每行一個問題）')
    batch_parser.add_argument('-o', '--output', type=str,
                            help='輸出文件（JSON格式）')
    
    # stats 命令
    subparsers.add_parser('stats', help='查看索引統計信息')
    
    args = parser.parse_args()
    
    # 執行命令
    if args.command == 'build':
        build_index(regenerate=args.regenerate)
    
    elif args.command == 'query':
        single_query(
            args.question,
            top_k_intents=args.top_k_intents,
            top_k_clauses=args.top_k_clauses,
            show_sources=not args.no_sources,
            show_details=args.details
        )
    
    elif args.command == 'interactive':
        interactive_mode()
    
    elif args.command == 'batch':
        batch_query(args.input, args.output)
    
    elif args.command == 'stats':
        show_stats()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()