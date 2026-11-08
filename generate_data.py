"""Generates the synthetic dataset: users.csv, feedback.csv and simulated_user_preferences.csv.

The simulated users have hidden preferences (archetypes such as career-focused or personality-focused). These
hidden weights are used only to simulate Accept/Reject behaviour and to evaluate the model; the model never sees them.
"""
import random, datetime as dt
import numpy as np, pandas as pd

from matching import MatchEngine

SEED = 42
rng = random.Random(SEED)
N_USERS = 120

MBTI_TYPES = ["INTJ","INTP","ENTJ","ENTP","INFJ","INFP","ENFJ","ENFP",
              "ISTJ","ISFJ","ESTJ","ESFJ","ISTP","ISFP","ESTP","ESFP"]

FAM = {
 "analytical": dict(
   interests=["chess","reading science books","solving logic puzzles","contributing to open-source","hiking","watching documentaries","playing the guitar","cycling"],
   values=["curiosity","accuracy","continuous learning","intellectual honesty","clear thinking"],
   edu=["B.Tech","M.Tech","M.Sc","B.Sc"]),
 "leader": dict(
   interests=["public speaking","listening to startup podcasts","running","mentoring students","travelling","strategy board games","volunteering","playing cricket"],
   values=["ownership","ambition","accountability","bold decisions","team growth"],
   edu=["B.Tech","MBA","BBA"]),
 "creative": dict(
   interests=["sketching","indie music","photography","writing short stories","travel blogging","visiting art galleries","cooking new cuisines","watching theatre"],
   values=["authenticity","creativity","empathy","freedom to experiment","storytelling"],
   edu=["B.Des","BA","MA","B.Tech"]),
 "people": dict(
   interests=["community volunteering","yoga","cooking for friends","reading psychology books","gardening","organising local meetups","classical dance","evening walks"],
   values=["kindness","trust","collaboration","inclusion","reliability"],
   edu=["MBA","MA","BBA","B.Sc"]),
 "hands": dict(
   interests=["building electronics","trekking","motorcycling","strategy gaming","playing football","fixing gadgets","playing badminton","running a home lab"],
   values=["practicality","independence","craftsmanship","efficiency","transparency"],
   edu=["B.Tech","BCA","B.Sc"]),
}

ROLES = [
 dict(title="Data Scientist", fam="analytical", mbti=["INTJ","INTP","ISTJ"],
      inds=["healthcare analytics","fintech","e-commerce analytics"],
      skills=["Python","SQL","machine learning","statistics","Tableau","A/B testing","forecasting"],
      ach=["built churn prediction models that reduced customer attrition","automated weekly reporting pipelines used by senior leadership","designed experiments that improved product conversion"],
      goals=[("grow into a lead data scientist role and mentor junior analysts","mentorship"),("collaborate on applied AI projects with product teams","collaboration"),("connect with data leaders and researchers in the industry","networking")]),
 dict(title="Machine Learning Engineer", fam="analytical", mbti=["INTP","INTJ","ISTP"],
      inds=["computer vision","conversational AI","recommendation systems"],
      skills=["PyTorch","TensorFlow","MLOps","Docker","feature engineering","model deployment","NLP"],
      ach=["shipped real-time models serving millions of requests","cut inference latency by half through model optimisation","set up monitoring for production ML pipelines"],
      goals=[("build production-grade AI products with a strong engineering team","collaboration"),("find a co-founder for an applied AI start-up","cofounder"),("learn from senior researchers about scaling ML systems","mentorship")]),
 dict(title="Backend Engineer", fam="analytical", mbti=["ISTJ","INTP","ISTP"],
      inds=["SaaS platforms","payments infrastructure","logistics technology"],
      skills=["Java","Spring Boot","PostgreSQL","microservices","REST APIs","Kafka","system design"],
      ach=["designed APIs handling thousands of transactions per second","migrated a monolith to microservices without downtime","mentored new joiners on clean coding practices"],
      goals=[("move into a staff engineer role focused on system design","mentorship"),("meet engineers who enjoy solving scalability problems","networking"),("collaborate on open-source developer tools","collaboration")]),
 dict(title="Financial Analyst", fam="analytical", mbti=["ISTJ","INTJ","ESTJ"],
      inds=["investment banking","corporate finance","insurance"],
      skills=["financial modelling","Excel","Power BI","SQL","risk analysis","valuation","budgeting"],
      ach=["built valuation models supporting multi-crore deals","streamlined monthly forecasting and cut reporting time","identified cost savings through variance analysis"],
      goals=[("transition into a data-driven strategy or product role","networking"),("learn from senior finance leaders about investment strategy","mentorship"),("collaborate with founders on fundraising and financial planning","collaboration")]),
 dict(title="Research Scientist", fam="analytical", mbti=["INTP","INTJ","INFJ"],
      inds=["natural language processing","bioinformatics","climate modelling"],
      skills=["Python","deep learning","research writing","statistics","LaTeX","experiment design","Transformers"],
      ach=["published peer-reviewed papers at international conferences","led a small research team on a funded project","released open datasets used by other labs"],
      goals=[("partner with industry teams to turn research into real products","collaboration"),("find thoughtful peers to discuss new research ideas","networking"),("guide early-career researchers through their first publications","mentorship")]),
 dict(title="Product Manager", fam="leader", mbti=["ENTJ","ENFJ","ENTP"],
      inds=["consumer mobile apps","B2B SaaS","edtech"],
      skills=["roadmapping","user research","SQL","stakeholder management","OKRs","go-to-market strategy","prioritisation"],
      ach=["launched features that doubled weekly active users","aligned engineering, design and sales around a shared roadmap","ran discovery interviews that reshaped the product vision"],
      goals=[("become a group product manager and build a strong product community","networking"),("find a technical co-founder for a new venture","cofounder"),("mentor aspiring product managers breaking into the field","mentorship")]),
 dict(title="Engineering Manager", fam="leader", mbti=["ENTJ","ESTJ","ENFJ"],
      inds=["cloud infrastructure","fintech","enterprise software"],
      skills=["team leadership","agile delivery","hiring","system architecture","project planning","coaching","cloud platforms"],
      ach=["scaled a team from five to twenty engineers","improved release reliability through better processes","coached engineers into senior and lead roles"],
      goals=[("grow into a director role and learn from experienced technology leaders","mentorship"),("exchange leadership lessons with other engineering managers","networking"),("partner with product leaders on ambitious platform initiatives","collaboration")]),
 dict(title="Startup Founder", fam="leader", mbti=["ENTJ","ENTP","ESTP"],
      inds=["health-tech","agri-tech","climate technology"],
      skills=["fundraising","business development","product strategy","hiring","pitching","market research","operations"],
      ach=["bootstrapped a venture to its first paying customers","raised a pre-seed round from angel investors","built a founding team from scratch"],
      goals=[("find a technical co-founder who shares a bold long-term vision","cofounder"),("meet mentors who have scaled companies before","mentorship"),("connect with investors and early customers","networking")]),
 dict(title="HR Business Partner", fam="people", mbti=["ENFJ","ESFJ","ISFJ"],
      inds=["technology services","retail","manufacturing"],
      skills=["talent acquisition","employee engagement","performance management","HR analytics","conflict resolution","onboarding","learning programs"],
      ach=["reduced attrition through better engagement programs","designed onboarding that helped new hires ramp up faster","introduced mentoring circles across departments"],
      goals=[("grow into a people-strategy leadership role","mentorship"),("meet professionals who care about healthy workplace culture","networking"),("collaborate on people-analytics projects","collaboration")]),
 dict(title="UX Designer", fam="creative", mbti=["INFP","ENFP","ISFP"],
      inds=["health-tech","fintech","learning platforms"],
      skills=["Figma","user research","prototyping","design systems","usability testing","interaction design","wireframing"],
      ach=["redesigned onboarding and lifted completion rates","built a design system adopted by five product teams","ran usability studies that shaped the product roadmap"],
      goals=[("collaborate with founders who care deeply about user experience","collaboration"),("find a mentor to grow into a design leadership role","mentorship"),("meet creative people to exchange feedback on design work","networking")]),
 dict(title="Content Strategist", fam="creative", mbti=["ENFP","INFP","INFJ"],
      inds=["digital media","SaaS marketing","non-profit communications"],
      skills=["copywriting","SEO","storytelling","editorial planning","content analytics","brand voice","social media"],
      ach=["grew organic traffic several times over with a content overhaul","launched a newsletter with a loyal readership","built editorial calendars that kept teams consistent"],
      goals=[("partner with start-ups that need a strong brand story","collaboration"),("meet designers and marketers for creative side projects","networking"),("learn from senior editors about building media brands","mentorship")]),
 dict(title="Brand Designer", fam="creative", mbti=["ISFP","INFP","ENFP"],
      inds=["lifestyle brands","consumer products","independent studios"],
      skills=["Adobe Illustrator","typography","brand identity","packaging design","Photoshop","motion graphics","art direction"],
      ach=["created identities for over twenty small businesses","led a rebrand that improved recognition and sales","won recognition at a regional design awards event"],
      goals=[("collaborate with founders who value thoughtful, original branding","collaboration"),("build friendships with fellow artists and makers","friendship"),("find an art director to learn from","mentorship")]),
 dict(title="DevOps Engineer", fam="hands", mbti=["ISTP","ISTJ","INTP"],
      inds=["cloud services","media streaming","banking technology"],
      skills=["Kubernetes","Terraform","AWS","CI/CD","Linux","monitoring","automation scripting"],
      ach=["cut deployment time from hours to minutes","built self-healing infrastructure that improved uptime","introduced infrastructure-as-code across teams"],
      goals=[("move into a site reliability leadership role","mentorship"),("meet engineers who enjoy building resilient systems","networking"),("collaborate on open-source infrastructure tools","collaboration")]),
 dict(title="Cybersecurity Analyst", fam="hands", mbti=["ISTP","INTJ","ISTJ"],
      inds=["banking security","government technology","cloud security"],
      skills=["penetration testing","SIEM tools","threat modelling","network security","Python scripting","incident response","compliance"],
      ach=["detected and contained several security incidents early","ran security awareness sessions across the company","closed critical vulnerabilities before public release"],
      goals=[("specialise in threat intelligence with guidance from senior practitioners","mentorship"),("share knowledge with other security professionals","networking"),("collaborate on security tools for small businesses","collaboration")]),
 dict(title="Sales and Business Development Manager", fam="leader", mbti=["ESTP","ENTJ","ESFP"],
      inds=["enterprise software","logistics services","financial services"],
      skills=["client relationships","negotiation","CRM tools","lead generation","account planning","presentations","market expansion"],
      ach=["exceeded annual revenue targets for three consecutive years","opened new regional markets from scratch","built long-term partnerships with key accounts"],
      goals=[("connect with founders who need help taking products to market","networking"),("find a partner to launch a new sales-driven venture","cofounder"),("learn strategy from senior commercial leaders","mentorship")]),
 dict(title="Marketing Manager", fam="creative", mbti=["ESFP","ENFP","ESFJ"],
      inds=["consumer goods","hospitality","digital brands"],
      skills=["campaign planning","brand strategy","Google Analytics","social media","influencer partnerships","event marketing","copywriting"],
      ach=["ran campaigns that tripled social media engagement","launched a product to a sold-out first month","built a community of loyal brand advocates"],
      goals=[("collaborate with creative teams on memorable brand campaigns","collaboration"),("expand my network among founders and creators","networking"),("grow into a chief marketing role with mentor support","mentorship")]),
 dict(title="Customer Success Manager", fam="people", mbti=["ESFJ","ISFJ","ENFJ"],
      inds=["B2B SaaS","edtech","health-tech"],
      skills=["customer onboarding","account management","churn analysis","CRM tools","training delivery","feedback loops","renewals"],
      ach=["raised customer retention through proactive outreach","built onboarding playbooks used by the whole team","turned at-risk accounts into long-term partners"],
      goals=[("move into a customer experience leadership role","mentorship"),("meet people who care about building lasting customer relationships","networking"),("collaborate with product teams on customer-driven features","collaboration")]),
 dict(title="Learning Experience Designer", fam="people", mbti=["INFJ","ENFJ","INFP"],
      inds=["education technology","corporate training","non-profit education"],
      skills=["instructional design","curriculum planning","e-learning tools","facilitation","learning analytics","storyboarding","coaching"],
      ach=["designed courses completed by thousands of learners","introduced project-based learning into a large programme","trained teachers to use digital tools with confidence"],
      goals=[("collaborate on meaningful education projects that widen access to learning","collaboration"),("find mentors in learning science and research","mentorship"),("connect with like-minded people who want to improve education","friendship")]),
 dict(title="Healthcare Operations Manager", fam="people", mbti=["ISFJ","ESTJ","ISTJ"],
      inds=["hospital administration","diagnostics","telemedicine"],
      skills=["process improvement","resource planning","compliance","vendor management","patient experience","data reporting","team coordination"],
      ach=["cut patient waiting times through better scheduling","introduced quality checks that improved audit results","coordinated staff across multiple departments smoothly"],
      goals=[("connect with technology people building tools for healthcare","networking"),("learn about health-tech from experienced founders","mentorship"),("collaborate on projects that make care more efficient","collaboration")]),
 dict(title="Innovation Consultant", fam="leader", mbti=["ENTP","ENFP","ENTJ"],
      inds=["management consulting","digital transformation","public sector innovation"],
      skills=["design thinking","workshop facilitation","market analysis","strategy","storytelling","stakeholder interviews","business modelling"],
      ach=["led innovation sprints that produced new revenue lines","advised leadership teams on digital strategy","ran hackathons that surfaced dozens of viable ideas"],
      goals=[("team up with founders who want fresh strategic thinking","collaboration"),("find a co-founder to launch a consulting-led product","cofounder"),("meet curious people from different industries","networking")]),
]

TRAIT = {
 "E": ["I thrive on meeting new people and sharing ideas out loud","I get energy from lively discussions and team brainstorming"],
 "I": ["I do my best thinking in quiet, focused stretches","I prefer deep one-on-one conversations over large gatherings"],
 "N": ["I like connecting patterns and imagining what could be built next","I am drawn to big-picture ideas and long-term possibilities"],
 "S": ["I like practical, hands-on solutions grounded in facts","I value proven methods and careful attention to detail"],
 "T": ["I make decisions using data and logic","I enjoy breaking problems down analytically"],
 "F": ["I lead with empathy and care about how people feel","I let my values and the people involved guide my choices"],
 "J": ["I plan ahead and like to finish what I start","I keep my work organised and deadline-driven"],
 "P": ["I stay flexible and adapt quickly when plans change","I enjoy exploring several options before committing"],
}
CITIES = {  # name: (lat, lon, sampling weight)
 "Bengaluru":(12.97,77.59,.24),"Hyderabad":(17.39,78.49,.14),"Pune":(18.52,73.86,.12),
 "Mumbai":(19.08,72.88,.12),"Delhi":(28.61,77.21,.12),"Chennai":(13.08,80.27,.08),
 "Kolkata":(22.57,88.36,.05),"Ahmedabad":(23.02,72.57,.04),"Jaipur":(26.91,75.79,.03),
 "Kochi":(9.93,76.27,.03),"Chandigarh":(30.73,76.78,.02),"Indore":(22.72,75.86,.01)}
FIRST_M = ["Aarav","Rohan","Karthik","Vikram","Arjun","Siddharth","Rahul","Nikhil","Aditya","Manish","Sanjay","Harsh","Varun","Pranav","Imran","Tarun","Gaurav","Deepak","Abhishek","Naveen","Kabir","Yash","Mohit","Suresh","Ashwin"]
FIRST_F = ["Ananya","Priya","Sneha","Kavya","Ishita","Meera","Divya","Neha","Pooja","Riya","Shreya","Aditi","Lakshmi","Nandini","Sara","Tanvi","Swati","Bhavana","Ritu","Anjali","Mahima","Pallavi","Sanjana","Rekha","Tara"]
LAST = ["Sharma","Iyer","Reddy","Patel","Nair","Gupta","Menon","Kulkarni","Joshi","Singh","Das","Rao","Verma","Bhat","Mehta","Chatterjee","Khan","Desai","Pillai","Agarwal","Banerjee","Kapoor","Shetty","Naidu"]

def make_users(n=N_USERS):
    rows, used = [], set()
    for i in range(n):
        role = ROLES[i % len(ROLES)]
        fam = FAM[role["fam"]]
        mbti = rng.choice(role["mbti"]) if rng.random() < 0.75 else rng.choice(MBTI_TYPES)
        gender = rng.choice(["Male","Female"])
        while True:
            name = f"{rng.choice(FIRST_M if gender=='Male' else FIRST_F)} {rng.choice(LAST)}"
            if name not in used: used.add(name); break
        yrs = rng.randint(1, 15)
        city = rng.choices(list(CITIES), weights=[c[2] for c in CITIES.values()])[0]
        lat, lon = CITIES[city][0] + rng.uniform(-.05,.05), CITIES[city][1] + rng.uniform(-.05,.05)
        industry = rng.choice(role["inds"]); s = rng.sample(role["skills"], 3)
        goal, looking = rng.choice(role["goals"])
        ints = rng.sample(fam["interests"], 3); vals = rng.sample(fam["values"], 2)
        prof = (f"{role['title']} with {yrs} year{'s' if yrs>1 else ''} of experience in {industry}, "
                f"skilled in {s[0]}, {s[1]}, and {s[2]}. Previously {rng.choice(role['ach'])}. "
                f"Looking to {goal}.")
        dims = rng.sample(range(4), 3)   # three of the four MBTI dimensions become personality sentences
        sents = [rng.choice(TRAIT[mbti[d]]) for d in sorted(dims)]
        about = (f"{sents[0]}. {sents[1]}. {sents[2]}. Outside work I enjoy {ints[0]}, {ints[1]}, and {ints[2]}. "
                 f"I value {vals[0]} and {vals[1]}.")
        rows.append(dict(
            user_id=f"U{i+1:03d}", name=name, age=22 + yrs + rng.randint(0,4), gender=gender,
            city=city, latitude=round(lat,4), longitude=round(lon,4),
            education=rng.choice(fam["edu"]), job_title=role["title"], industry=industry,
            years_experience=yrs, skills=", ".join(s), mbti=mbti,
            work_mode=rng.choice(["Remote","Hybrid","On-site"]), looking_for=looking,
            career_goal=goal[0].upper()+goal[1:], interests=", ".join(ints), values=", ".join(vals),
            professional_summary=prof, about_me=about,
            join_date=(dt.date(2024,10,1)+dt.timedelta(days=rng.randint(0,90))).isoformat()))
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True).assign(
        user_id=lambda d: [f"U{i+1:03d}" for i in range(len(d))])


ARCH = {"career-focused": (.80, .10, .10), "personality-focused": (.10, .80, .10),
        "proximity-focused": (.10, .10, .80), "balanced": (.40, .30, .30)}


def hidden_preferences(FEAT, seed=7):
    """Hidden per-user weights and the true probability that user i accepts user j."""
    sim_rng = np.random.default_rng(seed)
    names, probs = list(ARCH), [.35, .30, .15, .20]
    arch = sim_rng.choice(names, size=len(FEAT), p=probs)
    true_w = np.array([sim_rng.dirichlet(np.array(ARCH[a]) * 60) for a in arch])
    s_true = np.einsum("ijk,ik->ij", FEAT, true_w)
    thr = np.array([np.percentile(np.delete(s_true[i], i), 72) for i in range(len(FEAT))])
    p_true = 1 / (1 + np.exp(-25 * (s_true - thr[:, None])))
    return sim_rng, arch, true_w, thr, p_true


def pair_features(users_csv="users.csv"):
    eng = MatchEngine(users_csv, None)
    return eng, np.stack([eng.features(eng.users.iloc[i]) for i in range(len(eng.users))])


if __name__ == "__main__":
    users = make_users()
    users.to_csv("users.csv", index=False)
    eng, FEAT = pair_features()
    sim_rng, arch, true_w, thr, p_true = hidden_preferences(FEAT)
    rows = []
    for i, u in enumerate(users.user_id):
        n = int(sim_rng.integers(8, 11))
        cands = sim_rng.choice([j for j in range(N_USERS) if j != i], size=n, replace=False)
        days = np.sort(sim_rng.integers(0, 90, size=n))
        for j, d in zip(cands, days):
            rows.append(dict(user_id=u, matched_user_id=users.user_id[j],
                             action=int(sim_rng.random() < p_true[i, j]),
                             timestamp=(dt.date(2025, 1, 1) + dt.timedelta(days=int(d))).isoformat()))
    fb = pd.DataFrame(rows)
    fb.to_csv("feedback.csv", index=False)
    pd.DataFrame({"user_id": users.user_id, "archetype": arch, "true_w1": true_w[:, 0].round(3),
                  "true_w2": true_w[:, 1].round(3), "true_w3": true_w[:, 2].round(3),
                  "accept_threshold": thr.round(3)}).to_csv("simulated_user_preferences.csv", index=False)
    assert not fb.duplicated().any() and fb.action.isin([0, 1]).all() and (fb.user_id != fb.matched_user_id).all()
    print(f"users: {users.shape} | feedback rows: {len(fb)} | accept rate: {fb.action.mean():.1%}")
