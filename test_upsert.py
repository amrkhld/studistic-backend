from app.services.supabase import get_supabase_admin
admin = get_supabase_admin()
print(admin.table("student_features").upsert)
