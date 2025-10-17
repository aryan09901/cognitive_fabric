#!/usr/bin/env python3
"""
Demo script showcasing Cognitive Fabric capabilities
"""

import asyncio
import time
import json
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.multi_agent.orchestrator import MultiAgentOrchestrator
from agents.core.cognitive_node import CognitiveNode
from monitoring.dashboard import RealTimeDashboard

class CognitiveFabricDemo:
    """
    Demo class to showcase Cognitive Fabric capabilities
    """
    
    def __init__(self):
        self.orchestrator = MultiAgentOrchestrator("0xDemoAddress")
        self.setup_demo_agents()
    
    def setup_demo_agents(self):
        """Setup demo agents with different specializations"""
        
        agent_configs = [
            {
                'agent_id': 'research_specialist',
                'config': {
                    'LLM_MODEL': 'mistralai/Mistral-7B-Instruct-v0.1',
                    'specialization': 'research',
                    'capabilities': ['deep_analysis', 'citation', 'verification']
                },
                'description': 'Specialized in research and academic content'
            },
            {
                'agent_id': 'technical_expert',
                'config': {
                    'LLM_MODEL': 'mistralai/Mistral-7B-Instruct-v0.1', 
                    'specialization': 'technical',
                    'capabilities': ['code_analysis', 'technical_explanation', 'problem_solving']
                },
                'description': 'Specialized in technical and programming topics'
            },
            {
                'agent_id': 'general_advisor',
                'config': {
                    'LLM_MODEL': 'mistralai/Mistral-7B-Instruct-v0.1',
                    'specialization': 'general',
                    'capabilities': ['broad_knowledge', 'explanation', 'summarization']
                },
                'description': 'General knowledge and advisory capabilities'
            }
        ]
        
        for agent_config in agent_configs:
            self.orchestrator.register_agent(
                agent_config['agent_id'],
                agent_config['config']
            )
            print(f"Registered agent: {agent_config['agent_id']} - {agent_config['description']}")
    
    async def run_demo_queries(self, queries: List[Dict[str, Any]]):
        """Run demo queries and display results"""
        
        print("\n" + "="*80)
        print("COGNITIVE FABRIC DEMO")
        print("="*80)
        
        for i, query_info in enumerate(queries, 1):
            print(f"\n📝 QUERY {i}: {query_info['query']}")
            print("-" * 60)
            
            start_time = time.time()
            
            try:
                if query_info.get('collaborative', False):
                    result = await self.orchestrator.collaborative_solving(
                        query_info['query'],
                        query_info.get('context', {})
                    )
                else:
                    result = await self.orchestrator.route_query(
                        query_info['query'],
                        query_info.get('agent_id'),
                        query_info.get('context', {})
                    )
                
                processing_time = time.time() - start_time
                
                # Display results
                print(f"🕒 Processing Time: {processing_time:.2f}s")
                print(f"✅ Verification Score: {result.get('verification_score', 0):.2f}")
                print(f"🎯 Confidence: {result.get('confidence', 0):.2f}")
                
                if 'agents_participated' in result:
                    print(f"👥 Agents Participated: {result['agents_participated']}")
                elif 'agents_used' in result:
                    print(f"🤖 Agent Used: {result['agents_used']}")
                
                print(f"\n💬 RESPONSE:")
                print(result['response'])
                
                if result.get('sources'):
                    print(f"\n📚 SOURCES ({len(result['sources'])}):")
                    for j, source in enumerate(result['sources'][:3], 1):  # Show top 3
                        print(f"  {j}. {source.get('content', '')[:100]}...")
                
                print(f"\n📊 METADATA:")
                for key, value in result.get('metadata', {}).items():
                    print(f"  {key}: {value}")
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
            
            print("-" * 60)
            await asyncio.sleep(1)  # Brief pause between queries
    
    async def showcase_collaboration(self):
        """Showcase multi-agent collaboration"""
        
        print("\n" + "🤝 MULTI-AGENT COLLABORATION DEMO")
        print("=" * 50)
        
        complex_query = """
        Analyze the impact of artificial intelligence on climate change from multiple perspectives: 
        technical capabilities for monitoring, ethical considerations in deployment, 
        and policy recommendations for sustainable AI development.
        """
        
        print(f"Complex Query: {complex_query.strip()}")
        
        start_time = time.time()
        result = await self.orchestrator.collaborative_solving(complex_query)
        processing_time = time.time() - start_time
        
        print(f"\nCollaborative Processing Time: {processing_time:.2f}s")
        print(f"Agents Participated: {result.get('agents_participated', 0)}")
        print(f"Overall Verification Score: {result.get('verification_score', 0):.2f}")
        
        print(f"\nCollaborative Response:")
        print(result['collaborative_response'])
        
        if result.get('individual_responses'):
            print(f"\nIndividual Agent Contributions:")
            for i, individual in enumerate(result['individual_responses'][:3], 1):
                print(f"Agent {i}: {individual.get('response', '')[:150]}...")
    
    async def run_performance_demo(self):
        """Run performance demonstration"""
        
        print("\n" + "🚀 PERFORMANCE DEMONSTRATION")
        print("=" * 40)
        
        # Test with multiple concurrent queries
        test_queries = [
            "Explain quantum computing basics",
            "What is blockchain technology?",
            "How do neural networks learn?",
            "Describe renewable energy sources",
            "What is CRISPR gene editing?"
        ]
        
        print(f"Running {len(test_queries)} concurrent queries...")
        
        start_time = time.time()
        
        tasks = []
        for query in test_queries:
            task = self.orchestrator.route_query(query)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if not isinstance(r, Exception))
        
        print(f"Completed {successful}/{len(test_queries)} queries in {total_time:.2f}s")
        print(f"Average time per query: {total_time/len(test_queries):.2f}s")
        print(f"Queries per second: {len(test_queries)/total_time:.2f}")
        
        # Show verification statistics
        verification_scores = [
            r.get('verification_score', 0) for r in results 
            if not isinstance(r, Exception)
        ]
        if verification_scores:
            avg_verification = sum(verification_scores) / len(verification_scores)
            print(f"Average verification score: {avg_verification:.2f}")

async def main():
    """Main demo execution"""
    
    demo = CognitiveFabricDemo()
    
    # Demo queries
    demo_queries = [
        {
            'query': 'What are the latest developments in fusion energy research?',
            'description': 'Research-focused query',
            'collaborative': False
        },
        {
            'query': 'Explain how attention mechanisms work in transformer models with code examples',
            'description': 'Technical query requiring code',
            'collaborative': False,
            'agent_id': 'technical_expert'
        },
        {
            'query': 'Compare and contrast blockchain consensus mechanisms: Proof of Work vs Proof of Stake',
            'description': 'Comparative analysis',
            'collaborative': True
        },
        {
            'query': 'What are the verified health benefits of intermittent fasting?',
            'description': 'Health information requiring verification',
            'collaborative': False,
            'context': {'domain': 'medical', 'verification_level': 'strict'}
        }
    ]
    
    # Run basic demo
    await demo.run_demo_queries(demo_queries)
    
    # Showcase collaboration
    await demo.showcase_collaboration()
    
    # Performance demo
    await demo.run_performance_demo()
    
    print("\n" + "="*80)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Display system summary
    print(f"\nSystem Summary:")
    print(f"• Total Agents: {len(demo.orchestrator.agents)}")
    print(f"• Agent Specializations: {list(demo.orchestrator.agents.keys())}")
    print(f"• Capabilities: RAG + RL + Blockchain Verification + Multi-Agent Collaboration")

if __name__ == "__main__":
    asyncio.run(main())