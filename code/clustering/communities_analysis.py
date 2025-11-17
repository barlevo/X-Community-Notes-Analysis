"""
Specialized Visualizations for Community Structure Analysis

This module creates detailed visualizations for community detection analysis,
including conflict topics analysis, diversity metrics, and topic specialists.

Author: Data Science Project Team
Date: 2024
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

class SpecializedVisualizationsDevide:
    """
    Creates specialized visualizations for community analysis.
    
    This class provides comprehensive visualization capabilities for community
    detection results, including conflict topic analysis, diversity metrics,
    topic specialists, and community leadership matrices.
    
    Attributes:
        results_dir: Directory for saving visualization outputs
        specialization_df: DataFrame with community specialization metrics
        community_topic_counts: DataFrame with topic counts per community
        community_topic_proportions: DataFrame with topic proportions per community
    """
    
    def __init__(self, results_dir: str | Path = "specialized_visualizations_devide",
                 analysis_dir: str | Path = "final_community_analysis_devide"):
        """
        Initialize the SpecializedVisualizationsDevide.
        
        Args:
            results_dir: Path to directory for saving visualizations
            analysis_dir: Path to directory containing community analysis results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.analysis_dir = Path(analysis_dir)
        
        # Load the analysis results
        self.load_analysis_results()
    
    def load_analysis_results(self) -> bool:
        """
        Load existing analysis results from community structure analysis.
        
        Returns:
            True if loading successful, False otherwise
            
        Raises:
            FileNotFoundError: If required analysis files are not found
        """
        print("📂 Loading devide analysis results...")
        
        try:
            self.specialization_df = pd.read_csv(self.analysis_dir / "specialization_analysis.csv")
            self.community_topic_counts = pd.read_csv(
                self.analysis_dir / "community_topic_counts.csv", index_col=0
            )
            self.community_topic_proportions = pd.read_csv(
                self.analysis_dir / "community_topic_proportions.csv", index_col=0
            )
            
            print(f"✅ Loaded analysis for {len(self.specialization_df)} communities")
            print(f"✅ Topic categories: {len(self.community_topic_proportions.columns)}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading analysis results: {e}")
            print("Please run the devide topic analysis first!")
            return False
        
        return True
    
    def create_conflict_topics_analysis(self) -> pd.Series:
        """
        Create specialized analysis for conflict-related topics.
        
        Analyzes which communities specialize in conflict topics and creates
        heatmaps and engagement rankings.
        
        Returns:
            Series with conflict engagement scores per community
        """
        print("\n🔥 Creating conflict topics analysis...")
        
        # Define conflict-related topics
        conflict_topics = ['UkraineConflict', 'GazaConflict', 'SyriaWar', 'OtherConflicts', 'Iran']
        available_conflict_topics = [topic for topic in conflict_topics if topic in self.community_topic_proportions.columns]
        
        if not available_conflict_topics:
            print("⚠️  No conflict topics found in data")
            return
        
        # Extract conflict topic data
        conflict_data = self.community_topic_proportions[available_conflict_topics]
        
        # Calculate conflict engagement score for each community
        conflict_engagement = conflict_data.sum(axis=1).sort_values(ascending=False)
        
        # Create conflict heatmap
        plt.figure(figsize=(14, 10))
        
        # Select top communities for conflict engagement
        top_conflict_communities = conflict_engagement.head(15).index
        heatmap_data = conflict_data.loc[top_conflict_communities]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Reds', 
                   cbar_kws={'label': 'Topic Proportion'}, linewidths=0.5)
        plt.title('Conflict Topics Specialization by Community (Devide Structure)\nTop 15 Communities by Conflict Engagement', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Conflict Topics', fontsize=12)
        plt.ylabel('Community ID', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.results_dir / "conflict_topics_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Conflict engagement ranking
        plt.figure(figsize=(12, 8))
        top_15 = conflict_engagement.head(15)
        bars = plt.bar([f"C{int(x)}" for x in top_15.index], top_15.values,
                      color='darkred', alpha=0.7)
        plt.xlabel('Community', fontsize=12)
        plt.ylabel('Total Conflict Engagement Score', fontsize=12)
        plt.title('Communities by Conflict Topics Engagement (Devide Structure)', 
                 fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "conflict_engagement_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🔥 Conflict analysis saved to {self.results_dir}")
        
        return conflict_engagement
    
    def create_diversity_analysis(self) -> None:
        """
        Create detailed diversity score analysis.
        
        Creates visualizations showing:
        - Diversity vs community size
        - Diversity score distribution
        """
        print("\n🌈 Creating diversity analysis...")
        
        # Diversity vs Size analysis
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(self.specialization_df['total_notes'], 
                            self.specialization_df['diversity_score'],
                            s=100, alpha=0.7, c=self.specialization_df['specialization_score'], 
                            cmap='RdYlBu')
        plt.colorbar(scatter, label='Specialization Score')
        plt.xlabel('Community Size (Number of Notes)', fontsize=12)
        plt.ylabel('Diversity Score', fontsize=12)
        plt.title('Community Diversity Analysis (Devide Structure)\nSize vs Diversity (Color = Specialization)', 
                 fontsize=16, fontweight='bold')
        
        # Annotate extreme cases
        for _, row in self.specialization_df.iterrows():
            if (row['diversity_score'] > 0.9 or row['diversity_score'] < 0.7 or 
                row['total_notes'] > 100000):
                plt.annotate(f"C{int(row['community_id'])}", 
                           (row['total_notes'], row['diversity_score']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "diversity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Diversity score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.specialization_df['diversity_score'], bins=15, alpha=0.7, 
                color='lightblue', edgecolor='black')
        plt.axvline(self.specialization_df['diversity_score'].mean(), color='red', 
                   linestyle='--', label=f'Mean: {self.specialization_df["diversity_score"].mean():.3f}')
        plt.xlabel('Diversity Score', fontsize=12)
        plt.ylabel('Number of Communities', fontsize=12)
        plt.title('Distribution of Community Diversity Scores (Devide Structure)', 
                 fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "diversity_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🌈 Diversity analysis saved to {self.results_dir}")
    
    def create_topic_specialists_analysis(self) -> Dict[str, pd.Series]:
        """
        Create analysis of topic specialists.
        
        Identifies top specialist communities for each topic and creates
        visualization grids.
        
        Returns:
            Dictionary mapping topic names to top specialist communities
        """
        print("\n🎯 Creating topic specialists analysis...")
        
        # For each topic, find the top specialist communities
        topic_specialists = {}
        
        for topic in self.community_topic_proportions.columns:
            # Get communities sorted by their specialization in this topic
            topic_data = self.community_topic_proportions[topic].sort_values(ascending=False)
            
            # Get top 3 specialists
            top_specialists = topic_data.head(3)
            topic_specialists[topic] = top_specialists
        
        # Create topic specialists grid
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle('Topic Specialists by Category (Devide Structure)\nTop 3 Communities per Topic', 
                    fontsize=16, fontweight='bold')
        
        topics = list(topic_specialists.keys())
        
        for i, topic in enumerate(topics[:15]):  # Limit to 15 topics
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            
            specialists = topic_specialists[topic]
            
            # Create bar chart for top specialists
            bars = ax.bar([f"C{int(x)}" for x in specialists.index], 
                         specialists.values, color='steelblue', alpha=0.7)
            
            ax.set_title(f'{topic}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Proportion', fontsize=8)
            ax.tick_params(axis='x', labelsize=8, rotation=45)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, specialists.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        
        # Hide empty subplots
        for i in range(len(topics), 15):
            row = i // 5
            col = i % 5
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "topic_specialists_grid.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎯 Topic specialists analysis saved to {self.results_dir}")
        
        return topic_specialists
    
    def create_comparative_radar_charts(self) -> None:
        """
        Create radar charts for top communities.
        
        Generates polar/radar charts showing topic profiles for the
        largest communities.
        """
        print("\n📡 Creating radar charts for top communities...")
        
        # Select top 6 communities by size for radar analysis
        top_communities = self.specialization_df.nlargest(6, 'total_notes')
        
        # Select top topics for radar (limit to 8 for readability)
        topic_sums = self.community_topic_proportions.sum().sort_values(ascending=False)
        top_topics = topic_sums.head(8).index.tolist()
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(top_topics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
        fig.suptitle('Community Topic Profiles - Top 6 Communities (Devide Structure)', 
                    fontsize=16, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for idx, (_, community_row) in enumerate(top_communities.iterrows()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            community_id = community_row['community_id']
            
            # Get topic proportions for this community
            if community_id in self.community_topic_proportions.index:
                values = self.community_topic_proportions.loc[community_id, top_topics].tolist()
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=f'Community {int(community_id)}', 
                       color=colors[idx])
                ax.fill(angles, values, alpha=0.25, color=colors[idx])
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(top_topics, fontsize=8)
                ax.set_ylim(0, max(max(values), 0.3))
                ax.set_title(f'Community {int(community_id)}\n{community_row["dominant_topic"]} Specialist\n'
                           f'{community_row["total_notes"]:,} notes', 
                           fontsize=10, fontweight='bold')
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "community_radar_charts.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📡 Radar charts saved to {self.results_dir}")
    
    def create_topic_dominance_analysis(self) -> pd.DataFrame:
        """
        Create topic dominance and competition analysis.
        
        Analyzes which communities dominate each topic and calculates
        competition levels.
        
        Returns:
            DataFrame with dominance and competition metrics per topic
        """
        print("\n👑 Creating topic dominance analysis...")
        
        # Calculate topic dominance metrics
        dominance_data = []
        
        for topic in self.community_topic_proportions.columns:
            topic_data = self.community_topic_proportions[topic].sort_values(ascending=False)
            
            # Top community's dominance
            top_community = topic_data.index[0]
            top_share = topic_data.iloc[0]
            
            # Competition level (entropy of top 3)
            top_3_shares = topic_data.head(3)
            if len(top_3_shares) >= 3 and top_3_shares.sum() > 0:
                normalized_shares = top_3_shares / top_3_shares.sum()
                competition_entropy = -sum(p * np.log2(p) for p in normalized_shares if p > 0)
                competition_level = competition_entropy / np.log2(3)  # Normalize to [0,1]
            else:
                competition_level = 0
            
            dominance_data.append({
                'topic': topic,
                'dominant_community': top_community,
                'dominance_score': top_share,
                'competition_level': competition_level,
                'total_engagement': self.community_topic_counts[topic].sum()
            })
        
        dominance_df = pd.DataFrame(dominance_data)
        
        # Topic dominance vs competition scatter
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(dominance_df['dominance_score'], 
                            dominance_df['competition_level'],
                            s=dominance_df['total_engagement']/100,
                            alpha=0.7, c=range(len(dominance_df)), cmap='tab20')
        
        plt.xlabel('Dominance Score (Top Community Share)', fontsize=12)
        plt.ylabel('Competition Level', fontsize=12)
        plt.title('Topic Dominance vs Competition (Devide Structure)\nBubble size = Total engagement', 
                 fontsize=16, fontweight='bold')
        
        # Annotate points
        for _, row in dominance_df.iterrows():
            plt.annotate(f"{row['topic']}\nC{int(row['dominant_community'])}", 
                        (row['dominance_score'], row['competition_level']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "topic_dominance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"👑 Topic dominance analysis saved to {self.results_dir}")
        
        return dominance_df
    
    def create_topic_leadership_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create topic leadership matrices (top 3 and top 5 communities per topic).
        
        Calculates leadership scores based on community rankings and topic proportions.
        Saves both CSV files and visualizations.
        
        Returns:
            Tuple of (leadership_matrix_top3, leadership_matrix_top5) DataFrames
        """
        print("\n👥 Creating topic leadership matrices...")
        
        # Create leadership matrix (like the original) - Top 3
        leadership_matrix_top3 = pd.DataFrame(index=self.community_topic_proportions.index, 
                                            columns=self.community_topic_proportions.columns)
        leadership_matrix_top3 = leadership_matrix_top3.fillna(0)
        
        for topic in self.community_topic_proportions.columns:
            top_3 = self.community_topic_proportions[topic].nlargest(3)
            for rank, (community, proportion) in enumerate(top_3.items()):
                # Score: 3 for 1st place, 2 for 2nd, 1 for 3rd (multiplied by proportion)
                leadership_matrix_top3.loc[community, topic] = (3 - rank) * proportion
        
        # Create leadership matrix - Top 5
        leadership_matrix_top5 = pd.DataFrame(index=self.community_topic_proportions.index, 
                                            columns=self.community_topic_proportions.columns)
        leadership_matrix_top5 = leadership_matrix_top5.fillna(0)
        
        for topic in self.community_topic_proportions.columns:
            top_5 = self.community_topic_proportions[topic].nlargest(5)
            for rank, (community, proportion) in enumerate(top_5.items()):
                # Score: 5 for 1st place, 4 for 2nd, 3 for 3rd, 2 for 4th, 1 for 5th
                leadership_matrix_top5.loc[community, topic] = (5 - rank) * proportion
        
        # Save the matrices as CSV
        leadership_matrix_top3.to_csv(self.results_dir / "topic_leadership_matrix_top3.csv")
        leadership_matrix_top5.to_csv(self.results_dir / "topic_leadership_matrix_top5.csv")
        
        # Create visualization for Top 3 Leadership Matrix (matching original style)
        plt.figure(figsize=(16, 10))
        im = plt.imshow(leadership_matrix_top3.values, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='Leadership Score (Position × Proportion)')
        plt.title('Topic Leadership Matrix - Top 3 Communities (Devide Structure)\n(Top 3 communities per topic)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Community ID', fontsize=12)
        
        plt.xticks(range(len(self.community_topic_proportions.columns)), 
                  self.community_topic_proportions.columns, rotation=45, ha='right')
        plt.yticks(range(len(self.community_topic_proportions.index)), 
                  [f"C{int(x)}" for x in self.community_topic_proportions.index])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "topic_leadership_matrix_top3.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create visualization for Top 5 Leadership Matrix
        plt.figure(figsize=(16, 10))
        im = plt.imshow(leadership_matrix_top5.values, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='Leadership Score (Position × Proportion)')
        plt.title('Topic Leadership Matrix - Top 5 Communities (Devide Structure)\n(Top 5 communities per topic)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Community ID', fontsize=12)
        
        plt.xticks(range(len(self.community_topic_proportions.columns)), 
                  self.community_topic_proportions.columns, rotation=45, ha='right')
        plt.yticks(range(len(self.community_topic_proportions.index)), 
                  [f"C{int(x)}" for x in self.community_topic_proportions.index])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "topic_leadership_matrix_top5.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"👥 Topic leadership matrices saved to {self.results_dir}")
        print(f"   📊 topic_leadership_matrix_top3.csv - Leadership scores (top 3)")
        print(f"   📊 topic_leadership_matrix_top5.csv - Leadership scores (top 5)")
        print(f"   🎨 topic_leadership_matrix_top3.png - Visual matrix (top 3)")
        print(f"   🎨 topic_leadership_matrix_top5.png - Visual matrix (top 5)")
        
        return leadership_matrix_top3, leadership_matrix_top5
    
    def create_summary_dashboard(self) -> None:
        """
        Create a comprehensive summary dashboard.
        
        Generates a multi-panel dashboard showing key community metrics
        including specialization, diversity, size, and topic engagement.
        """
        print("\n📊 Creating summary dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Top specialists (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        top_specialists = self.specialization_df.head(8)
        bars1 = ax1.bar([f"C{int(x)}" for x in top_specialists['community_id']], 
                       top_specialists['specialization_score'],
                       color='steelblue', alpha=0.7)
        ax1.set_title('Top Specialized Communities', fontweight='bold')
        ax1.set_ylabel('Specialization Score')
        ax1.grid(True, alpha=0.3)
        
        # 2. Diversity distribution (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.hist(self.specialization_df['diversity_score'], bins=10, alpha=0.7, 
                color='lightgreen', edgecolor='black')
        ax2.set_title('Diversity Score Distribution', fontweight='bold')
        ax2.set_xlabel('Diversity Score')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # 3. Size vs Specialization (middle-left)
        ax3 = fig.add_subplot(gs[1, :2])
        scatter = ax3.scatter(self.specialization_df['total_notes'], 
                             self.specialization_df['specialization_score'],
                             alpha=0.7, c=self.specialization_df['diversity_score'], 
                             cmap='RdYlBu_r')
        ax3.set_xlabel('Community Size')
        ax3.set_ylabel('Specialization Score')
        ax3.set_title('Size vs Specialization', fontweight='bold')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Topic engagement (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        topic_totals = self.community_topic_counts.sum().sort_values(ascending=False).head(8)
        ax4.barh(range(len(topic_totals)), topic_totals.values, color='orange', alpha=0.7)
        ax4.set_yticks(range(len(topic_totals)))
        ax4.set_yticklabels(topic_totals.index, fontsize=9)
        ax4.set_title('Top Topics by Engagement', fontweight='bold')
        ax4.set_xlabel('Total Notes')
        ax4.grid(True, alpha=0.3)
        
        # 5. Community sizes (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        largest_communities = self.specialization_df.nlargest(10, 'total_notes')
        bars5 = ax5.bar([f"C{int(row['community_id'])}\n{row['dominant_topic'][:8]}" 
                        for _, row in largest_communities.iterrows()], 
                       largest_communities['total_notes'],
                       color='purple', alpha=0.7)
        ax5.set_title('Largest Communities by Size', fontweight='bold')
        ax5.set_ylabel('Number of Notes')
        ax5.tick_params(axis='x', rotation=45, labelsize=9)
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Community Analysis Dashboard (Devide Structure)', 
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / "summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary dashboard saved to {self.results_dir}")
    
    def run_all_visualizations(self) -> Dict[str, Any]:
        """
        Run all specialized visualizations.
        
        Executes the complete visualization pipeline:
        1. Conflict topics analysis
        2. Diversity analysis
        3. Topic specialists analysis
        4. Comparative radar charts
        5. Topic dominance analysis
        6. Topic leadership matrices
        7. Summary dashboard
        
        Returns:
            Dictionary containing all analysis results
        """
        print("🚀 CREATING SPECIALIZED VISUALIZATIONS (DEVIDE STRUCTURE)")
        print("=" * 70)
        
        if not hasattr(self, 'specialization_df'):
            print("❌ Analysis results not loaded. Please run devide topic analysis first!")
            return
        
        # Create all visualizations
        conflict_engagement = self.create_conflict_topics_analysis()
        self.create_diversity_analysis()
        topic_specialists = self.create_topic_specialists_analysis()
        self.create_comparative_radar_charts()
        dominance_df = self.create_topic_dominance_analysis()
        leadership_top3, leadership_top5 = self.create_topic_leadership_matrix()
        self.create_summary_dashboard()
        
        print(f"\n✅ All specialized visualizations created!")
        print(f"📁 Results saved to: {self.results_dir}")
        print("\nGenerated visualizations:")
        print("  🔥 conflict_topics_heatmap.png - Conflict topics analysis")
        print("  🔥 conflict_engagement_ranking.png - Conflict engagement ranking")
        print("  🌈 diversity_analysis.png - Diversity vs size analysis")
        print("  🌈 diversity_distribution.png - Diversity score distribution")
        print("  🎯 topic_specialists_grid.png - Topic specialists by category")
        print("  📡 community_radar_charts.png - Community topic profiles")
        print("  👑 topic_dominance_analysis.png - Topic dominance vs competition")
        print("  👥 topic_leadership_matrix_top3.png - Topic leadership (top 3)")
        print("  👥 topic_leadership_matrix_top5.png - Topic leadership (top 5)")
        print("  📊 summary_dashboard.png - Comprehensive dashboard")
        
        return {
            'conflict_engagement': conflict_engagement,
            'topic_specialists': topic_specialists,
            'dominance_analysis': dominance_df,
            'leadership_top3': leadership_top3,
            'leadership_top5': leadership_top5
        }

def main() -> Optional[Dict[str, Any]]:
    """
    Main execution function.
    
    Returns:
        Dictionary containing visualization results, or None if execution fails
    """
    visualizer = SpecializedVisualizationsDevide()
    results = visualizer.run_all_visualizations()
    
    if results:
        print("\n🎯 KEY INSIGHTS FROM SPECIALIZED VISUALIZATIONS:")
        print("-" * 50)
        
        # Top conflict-engaged community
        if 'conflict_engagement' in results and len(results['conflict_engagement']) > 0:
            top_conflict = results['conflict_engagement'].index[0]
            top_conflict_score = results['conflict_engagement'].iloc[0]
            print(f"Most conflict-engaged: Community {top_conflict} "
                  f"(engagement score: {top_conflict_score:.3f})")
        
        # Most dominated topic
        if 'dominance_analysis' in results:
            most_dominated = results['dominance_analysis'].loc[results['dominance_analysis']['dominance_score'].idxmax()]
            print(f"Most dominated topic: {most_dominated['topic']} "
                  f"by Community {most_dominated['dominant_community']} "
                  f"({most_dominated['dominance_score']:.1%} share)")
        
        # Show a sample from leadership matrices
        if 'leadership_top3' in results:
            print(f"\nSample topic leadership (Politics):")
            if 'Politics' in results['leadership_top3'].columns:
                politics_data = results['leadership_top3']['Politics'].nlargest(3)
                for i, (community, score) in enumerate(politics_data.items()):
                    print(f"  {i+1}: Community {community} (leadership score: {score:.3f})")

if __name__ == "__main__":
    main()
