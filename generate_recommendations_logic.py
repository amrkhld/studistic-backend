def _generate_recommendations(features: dict, predicted_score: float) -> list[dict]:
    """Generate personalized recommendations based on student features."""
    recs = []
    
    # 1. Attendance (Most critical)
    attendance = features.get("attendance", 0)
    if attendance < 80:
        recs.append({
            "title": "Critically Improve Attendance",
            "description": f"Current: {attendance}% → Target: 90%+. Poor attendance strongly correlates with lower scores.",
            "priority": "high", "icon": "🚨"
        })
    elif attendance < 90:
        recs.append({
            "title": "Improve Attendance",
            "description": f"Current: {attendance}% → Target: 90%+. Every class counts towards your final grade.",
            "priority": "medium", "icon": "📌"
        })

    # 2. Hours Studied
    hours = features.get("hours_studied", 0)
    if hours < 15:
        recs.append({
            "title": "Increase Study Hours",
            "description": f"Current: {hours}h/wk → Target: 25h+. Dedicated study time is essential.",
            "priority": "high", "icon": "📚"
        })
    elif hours < 25:
        recs.append({
            "title": "Optimize Study Time",
            "description": f"Current: {hours}h/wk. Try adding 5 more hours per week for better retention.",
            "priority": "medium", "icon": "⏳"
        })

    # 3. Previous Scores
    prev_scores = features.get("previous_scores", 0)
    if prev_scores < 65:
        recs.append({
            "title": "Focus on Foundational Concepts",
            "description": f"Previous scores ({prev_scores}) indicate gaps. Review past materials before advancing.",
            "priority": "high", "icon": "🎯"
        })

    # 4. Tutoring
    if features.get("tutoring_sessions", 0) == 0 and predicted_score < 75:
        recs.append({
            "title": "Start Tutoring Sessions",
            "description": "Students attending even 1-2 tutoring sessions per month score significantly higher.",
            "priority": "medium", "icon": "👨‍🏫"
        })

    # 5. Sleep Schedule
    sleep = features.get("sleep_hours", 7)
    if sleep < 6:
        recs.append({
            "title": "Increase Sleep",
            "description": f"Current: {sleep}h. Lack of sleep impairs memory and cognitive function. Aim for 7-8h.",
            "priority": "high", "icon": "😴"
        })
    elif sleep > 9:
        recs.append({
            "title": "Regulate Sleep Schedule",
            "description": f"Current: {sleep}h. Oversleeping can cause lethargy. Stick to a consistent 7-8h schedule.",
            "priority": "medium", "icon": "⏰"
        })
    else:
        recs.append({
            "title": "Maintain Sleep Routine",
            "description": "Your 7-8h sleep schedule is optimal for academic performance.",
            "priority": "low", "icon": "🌙"
        })

    # 6. Motivation
    if features.get("motivation_level") == "Low":
        recs.append({
            "title": "Set Clear Goals",
            "description": "Low motivation can be overcome by breaking tasks into smaller, manageable milestones.",
            "priority": "high", "icon": "🔥"
        })

    # 7. Peer Influence
    if features.get("peer_influence") == "Negative":
        recs.append({
            "title": "Form a Study Group",
            "description": "Surround yourself with academically focused peers to positively influence your habits.",
            "priority": "medium", "icon": "👥"
        })

    # 8. Teacher Quality / Resources
    if features.get("teacher_quality") == "Low" or features.get("access_to_resources") == "Low":
        recs.append({
            "title": "Seek Alternative Resources",
            "description": "Utilize online courses, YouTube lectures, or library books to supplement your classes.",
            "priority": "medium", "icon": "🌐"
        })

    # 9. Physical Activity
    if features.get("physical_activity", 0) < 2:
        recs.append({
            "title": "Add Physical Activity",
            "description": "Exercise improves focus and reduces stress. Aim for at least 2-3 hours per week.",
            "priority": "low", "icon": "🏃"
        })

    # 10. Extracurriculars
    if features.get("extracurricular_activities") is False:
        recs.append({
            "title": "Balance with Extracurriculars",
            "description": "Engaging in clubs or sports can improve time management and mental well-being.",
            "priority": "low", "icon": "🎨"
        })

    # Sort recommendations: high -> medium -> low
    priority_map = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda x: priority_map[x["priority"]])
    
    # Take top 5 and assign IDs
    top_recs = recs[:5]
    for i, r in enumerate(top_recs):
        r["id"] = i + 1
        
    return top_recs
